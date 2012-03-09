/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.hadoop.hdfs.server.datanode;

import java.io.File;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.DFSTestUtil;
import org.apache.hadoop.hdfs.MiniDFSCluster;
import org.apache.hadoop.hdfs.server.namenode.FSNamesystem;
import org.apache.hadoop.hdfs.HdfsConfiguration;
import org.apache.hadoop.hdfs.DFSConfigKeys;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.logging.impl.Log4JLogger;
import org.apache.log4j.Level;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

/**
 * Test the ability of a DN to tolerate volume failures.
 */
public class TestDataNodeVolumeFailureToleration {

  private static final Log LOG = LogFactory.getLog(TestDataNodeVolumeFailureToleration.class);
  {
    ((Log4JLogger)TestDataNodeVolumeFailureToleration.LOG).getLogger().setLevel(Level.ALL);
  }

  private FileSystem fs;
  private MiniDFSCluster cluster;
  private Configuration conf;
  private String dataDir;

  // Sleep at least 3 seconds (a 1s heartbeat plus padding) to allow
  // for heartbeats to propagate from the datanodes to the namenode.
  final int WAIT_FOR_HEARTBEATS = 3000;

  // Wait at least (2 * re-check + 10 * heartbeat) seconds for
  // a datanode to be considered dead by the namenode.  
  final int WAIT_FOR_DEATH = 15000;

  @Before
  public void setUp() throws Exception {
    conf = new HdfsConfiguration();
    conf.setLong(DFSConfigKeys.DFS_BLOCK_SIZE_KEY, 512L);
    /*
     * Lower the DN heartbeat, DF rate, and recheck interval to one second
     * so state about failures and datanode death propagates faster.
     */
    conf.setInt(DFSConfigKeys.DFS_HEARTBEAT_INTERVAL_KEY, 1);
    conf.setInt(DFSConfigKeys.DFS_DF_INTERVAL_KEY, 1000);
    conf.setInt(DFSConfigKeys.DFS_NAMENODE_HEARTBEAT_RECHECK_INTERVAL_KEY, 1000);
    // Allow a single volume failure (there are two volumes)
    conf.setInt(DFSConfigKeys.DFS_DATANODE_FAILED_VOLUMES_TOLERATED_KEY, 1);
    cluster = new MiniDFSCluster.Builder(conf).numDataNodes(1).build();
    cluster.waitActive();
    fs = cluster.getFileSystem();
    dataDir = cluster.getDataDirectory();
  }

  @After
  public void tearDown() throws Exception {
    for (int i = 0; i < 3; i++) {
      new File(dataDir, "data"+(2*i+1)).setExecutable(true);
      new File(dataDir, "data"+(2*i+2)).setExecutable(true);
    }
    cluster.shutdown();
  }

  /**
   * Test the DFS_DATANODE_FAILED_VOLUMES_TOLERATED_KEY configuration
   * option, ie the DN shuts itself down when the number of failures
   * experienced drops below the tolerated amount.
   */
  @Test
  public void testConfigureMinValidVolumes() throws Exception {
    assumeTrue(!System.getProperty("os.name").startsWith("Windows"));

    // Bring up two additional datanodes that need both of their volumes
    // functioning in order to stay up.
    conf.setInt(DFSConfigKeys.DFS_DATANODE_FAILED_VOLUMES_TOLERATED_KEY, 0);
    cluster.startDataNodes(conf, 2, true, null, null);
    cluster.waitActive();
    FSNamesystem ns = cluster.getNamesystem();
    long origCapacity = DFSTestUtil.getLiveDatanodeCapacity(ns);
    long dnCapacity = DFSTestUtil.getDatanodeCapacity(ns, 0);

    // Fail a volume on the 2nd DN
    File dn2Vol1 = new File(dataDir, "data"+(2*1+1));
    assertTrue("Couldn't chmod local vol", dn2Vol1.setExecutable(false));

    // Should only get two replicas (the first DN and the 3rd)
    Path file1 = new Path("/test1");
    DFSTestUtil.createFile(fs, file1, 1024, (short)3, 1L);
    DFSTestUtil.waitReplication(fs, file1, (short)2);

    // Check that this single failure caused a DN to die.
    DFSTestUtil.waitForDatanodeStatus(ns, 2, 1, 0, 
        origCapacity - (1*dnCapacity), WAIT_FOR_HEARTBEATS);

    // If we restore the volume we should still only be able to get
    // two replicas since the DN is still considered dead.
    assertTrue("Couldn't chmod local vol", dn2Vol1.setExecutable(true));
    Path file2 = new Path("/test2");
    DFSTestUtil.createFile(fs, file2, 1024, (short)3, 1L);
    DFSTestUtil.waitReplication(fs, file2, (short)2);
  }

  /**
   * Test invalid DFS_DATANODE_FAILED_VOLUMES_TOLERATED_KEY values.
   */
  @Test
  public void testInvalidFailedVolumesConfig() throws Exception {
    assumeTrue(!System.getProperty("os.name").startsWith("Windows"));

    /*
     * Bring up another datanode that has an invalid value set.
     * We should still be able to create a file with two replicas
     * since the minimum valid volume parameter is only checked
     * when we experience a disk error.
     */
    conf.setInt(DFSConfigKeys.DFS_DATANODE_FAILED_VOLUMES_TOLERATED_KEY, -1);
    cluster.startDataNodes(conf, 1, true, null, null);
    cluster.waitActive();
    Path file1 = new Path("/test1");
    DFSTestUtil.createFile(fs, file1, 1024, (short)2, 1L);
    DFSTestUtil.waitReplication(fs, file1, (short)2);

    // Ditto if the value is too big.
    conf.setInt(DFSConfigKeys.DFS_DATANODE_FAILED_VOLUMES_TOLERATED_KEY, 100);
    cluster.startDataNodes(conf, 1, true, null, null);
    cluster.waitActive();
    Path file2 = new Path("/test1");
    DFSTestUtil.createFile(fs, file2, 1024, (short)2, 1L);
    DFSTestUtil.waitReplication(fs, file2, (short)2);
  }
}

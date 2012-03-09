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

package org.apache.hadoop.cli;

import org.apache.hadoop.cli.util.CLITestData.TestCmd;
import org.apache.hadoop.cli.util.CommandExecutor.Result;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.DFSConfigKeys;
import org.apache.hadoop.hdfs.DistributedFileSystem;
import org.apache.hadoop.hdfs.HDFSPolicyProvider;
import org.apache.hadoop.hdfs.MiniDFSCluster;
import org.apache.hadoop.security.authorize.PolicyProvider;
import org.junit.After;
import static org.junit.Assert.assertTrue;
import org.junit.Before;
import org.junit.Test;

public class TestHDFSCLI extends CLITestHelper {

  protected MiniDFSCluster dfsCluster = null;
  protected DistributedFileSystem dfs = null;
  protected String namenode = null;

  @Before
  @Override
  public void setUp() throws Exception {
    super.setUp();
    conf.setClass(PolicyProvider.POLICY_PROVIDER_CONFIG,
        HDFSPolicyProvider.class, PolicyProvider.class);
    
    // Many of the tests expect a replication value of 1 in the output
    conf.setInt("dfs.replication", 1);
    
    // Build racks and hosts configuration to test dfsAdmin -printTopology
    String [] racks =  {"/rack1", "/rack1", "/rack2", "/rack2",
                        "/rack2", "/rack3", "/rack4", "/rack4" };
    String [] hosts = {"host1", "host2", "host3", "host4",
                       "host5", "host6", "host7", "host8" };

    FileSystem fs;
    namenode = System.getProperty("test.cli.fs.default.name");
    if (namenode == null) {
      // Start up the mini dfs cluster
      dfsCluster = new MiniDFSCluster.Builder(conf).numDataNodes(8)
                                                   .racks(racks)
                                                   .hosts(hosts)
                                                   .build();
      namenode = conf.get(DFSConfigKeys.FS_DEFAULT_NAME_KEY, "file:///");
      fs = dfsCluster.getFileSystem();
    } else {
      conf.set(DFSConfigKeys.FS_DEFAULT_NAME_KEY, namenode);
      fs = FileSystem.get(conf);
    }

    username = System.getProperty("user.name");

    assertTrue("Not a HDFS: "+fs.getUri(),
               fs instanceof DistributedFileSystem);
    dfs = (DistributedFileSystem) fs;
  }

  @Override
  protected String getTestFile() {
    return "testHDFSConf.xml";
  }
  
  @After
  @Override
  public void tearDown() throws Exception {
    dfs.delete(new Path(testDirAbsolute), true);
    if (dfsCluster != null) {
      boolean success = false;
      dfs.close();
      dfsCluster.shutdown();
      success = true;
      Thread.sleep(2000);
      assertTrue("Error tearing down Mini DFS cluster", success);
      super.tearDown();
    }
  }

  @Override
  protected String expandCommand(final String cmd) {
    String expCmd = cmd;
    expCmd = expCmd.replaceAll("NAMENODE", namenode);
    expCmd = super.expandCommand(cmd);
    return expCmd;
  }
  
  @Override
  protected Result execute(TestCmd cmd) throws Exception {
    return CmdFactoryDFS.getCommandExecutor(cmd, namenode).executeCommand(cmd.getCmd());
  }

  @Test
  @Override
  public void testAll () {
    super.testAll();
  }
}

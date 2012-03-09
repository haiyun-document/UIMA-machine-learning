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
package org.apache.hadoop.hdfs;

import java.io.IOException;
import java.net.URL;

import junit.framework.TestCase;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.ChecksumException;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.Path;

/**
 * The test makes sure that NameNode detects presense blocks that do not have
 * any valid replicas. In addition, it verifies that HDFS front page displays
 * a warning in such a case.
 */
public class TestMissingBlocksAlert extends TestCase {
  
  private static final Log LOG = 
                           LogFactory.getLog(TestMissingBlocksAlert.class);
  
  public void testMissingBlocksAlert() throws IOException, 
                                       InterruptedException {
    
    MiniDFSCluster cluster = null;
    
    try {
      Configuration conf = new HdfsConfiguration();
      //minimize test delay
      conf.setInt(DFSConfigKeys.DFS_NAMENODE_REPLICATION_INTERVAL_KEY, 0);
      int fileLen = 10*1024;

      //start a cluster with single datanode
      cluster = new MiniDFSCluster.Builder(conf).build();
      cluster.waitActive();

      DistributedFileSystem dfs = 
                            (DistributedFileSystem) cluster.getFileSystem();

      // create a normal file
      DFSTestUtil.createFile(dfs, new Path("/testMissingBlocksAlert/file1"), 
                             fileLen, (short)3, 0);

      Path corruptFile = new Path("/testMissingBlocks/corruptFile");
      DFSTestUtil.createFile(dfs, corruptFile, fileLen, (short)3, 0);


      // Corrupt the block
      String block = DFSTestUtil.getFirstBlock(dfs, corruptFile).getBlockName();
      assertTrue(cluster.corruptReplica(block, 0));

      // read the file so that the corrupt block is reported to NN
      FSDataInputStream in = dfs.open(corruptFile); 
      try {
        in.readFully(new byte[fileLen]);
      } catch (ChecksumException ignored) { // checksum error is expected.      
      }
      in.close();

      LOG.info("Waiting for missing blocks count to increase...");

      while (dfs.getMissingBlocksCount() <= 0) {
        Thread.sleep(100);
      }
      assertTrue(dfs.getMissingBlocksCount() == 1);


      // Now verify that it shows up on webui
      URL url = new URL("http://" + conf.get(DFSConfigKeys.DFS_NAMENODE_HTTP_ADDRESS_KEY) + 
                        "/dfshealth.jsp");
      String dfsFrontPage = DFSTestUtil.urlGet(url);
      String warnStr = "WARNING : There are ";
      assertTrue("HDFS Front page does not contain expected warning", 
                 dfsFrontPage.contains(warnStr + "1 missing blocks"));

      // now do the reverse : remove the file expect the number of missing 
      // blocks to go to zero

      dfs.delete(corruptFile, true);

      LOG.info("Waiting for missing blocks count to be zero...");
      while (dfs.getMissingBlocksCount() > 0) {
        Thread.sleep(100);
      }

      // and make sure WARNING disappears
      // Now verify that it shows up on webui
      dfsFrontPage = DFSTestUtil.urlGet(url);
      assertFalse("HDFS Front page contains unexpected warning", 
                  dfsFrontPage.contains(warnStr));
    } finally {
      if (cluster != null) {
        cluster.shutdown();
      }
    }
  }
}

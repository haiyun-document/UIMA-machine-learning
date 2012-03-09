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
package org.apache.hadoop.hdfs.server.namenode;

import java.util.Collection;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.DFSTestUtil;
import org.apache.hadoop.hdfs.HdfsConfiguration;
import org.apache.hadoop.hdfs.MiniDFSCluster;
import org.apache.hadoop.hdfs.MiniDFSCluster.DataNodeProperties;
import org.apache.hadoop.hdfs.protocol.Block;
import org.apache.hadoop.hdfs.server.namenode.FSNamesystem.NumberReplicas;

import junit.framework.TestCase;

/**
 * Test if live nodes count per node is correct 
 * so NN makes right decision for under/over-replicated blocks
 */
public class TestNodeCount extends TestCase {
  public void testNodeCount() throws Exception {
    // start a mini dfs cluster of 2 nodes
    final Configuration conf = new HdfsConfiguration();
    final short REPLICATION_FACTOR = (short)2;
    final MiniDFSCluster cluster = 
      new MiniDFSCluster.Builder(conf).numDataNodes(REPLICATION_FACTOR).build();
    try {
      final FSNamesystem namesystem = cluster.getNamesystem();
      final FileSystem fs = cluster.getFileSystem();
      
      // populate the cluster with a one block file
      final Path FILE_PATH = new Path("/testfile");
      DFSTestUtil.createFile(fs, FILE_PATH, 1L, REPLICATION_FACTOR, 1L);
      DFSTestUtil.waitReplication(fs, FILE_PATH, REPLICATION_FACTOR);
      Block block = DFSTestUtil.getFirstBlock(fs, FILE_PATH);

      // keep a copy of all datanode descriptor
      DatanodeDescriptor[] datanodes = 
         namesystem.heartbeats.toArray(new DatanodeDescriptor[REPLICATION_FACTOR]);
      
      // start two new nodes
      cluster.startDataNodes(conf, 2, true, null, null);
      cluster.waitActive();
      
      // bring down first datanode
      DatanodeDescriptor datanode = datanodes[0];
      DataNodeProperties dnprop = cluster.stopDataNode(datanode.getName());
      // make sure that NN detects that the datanode is down
      synchronized (namesystem.heartbeats) {
        datanode.setLastUpdate(0); // mark it dead
        namesystem.heartbeatCheck();
      }
      // the block will be replicated
      DFSTestUtil.waitReplication(fs, FILE_PATH, REPLICATION_FACTOR);

      // restart the first datanode
      cluster.restartDataNode(dnprop);
      cluster.waitActive();
      
      // check if excessive replica is detected
      NumberReplicas num = null;
      do {
       synchronized (namesystem) {
         num = namesystem.blockManager.countNodes(block);
       }
      } while (num.excessReplicas() == 0);
      
      // find out a non-excess node
      Iterator<DatanodeDescriptor> iter = namesystem.blockManager.blocksMap.nodeIterator(block);
      DatanodeDescriptor nonExcessDN = null;
      while (iter.hasNext()) {
        DatanodeDescriptor dn = iter.next();
        Collection<Block> blocks = namesystem.blockManager.excessReplicateMap.get(dn.getStorageID());
        if (blocks == null || !blocks.contains(block) ) {
          nonExcessDN = dn;
          break;
        }
      }
      assertTrue(nonExcessDN!=null);
      
      // bring down non excessive datanode
      dnprop = cluster.stopDataNode(nonExcessDN.getName());
      // make sure that NN detects that the datanode is down
      synchronized (namesystem.heartbeats) {
        nonExcessDN.setLastUpdate(0); // mark it dead
        namesystem.heartbeatCheck();
      }
      
      // The block should be replicated
      do {
        namesystem.writeLock();
        num = namesystem.blockManager.countNodes(block);
        namesystem.writeUnlock();
      } while (num.liveReplicas() != REPLICATION_FACTOR);
      
      // restart the first datanode
      cluster.restartDataNode(dnprop);
      cluster.waitActive();
      
      // check if excessive replica is detected
      do {
        namesystem.writeLock();
        num = namesystem.blockManager.countNodes(block);
        namesystem.writeUnlock();
      } while (num.excessReplicas() != 2);
    } finally {
      cluster.shutdown();
    }
  }
}

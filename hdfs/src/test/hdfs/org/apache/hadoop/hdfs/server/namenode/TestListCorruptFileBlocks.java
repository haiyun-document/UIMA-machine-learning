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

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collection;
import java.util.Random;

import junit.framework.TestCase;

import org.apache.commons.logging.Log;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.hdfs.BlockMissingException;
import org.apache.hadoop.hdfs.DFSTestUtil;
import org.apache.hadoop.hdfs.HdfsConfiguration;
import org.apache.hadoop.hdfs.MiniDFSCluster;

/**
 * This class tests the listCorruptFileBlocks API.
 * We create 3 files; intentionally delete their blocks
 * Use listCorruptFileBlocks to validate that we get the list of corrupt
 * files/blocks; also test the "paging" support by calling the API
 * with a block # from a previous call and validate that the subsequent
 * blocks/files are also returned.
 */
public class TestListCorruptFileBlocks extends TestCase {
  static Log LOG = NameNode.stateChangeLog;

  /** check if nn.getCorruptFiles() returns a file that has corrupted blocks */
  public void testListCorruptFilesCorruptedBlock() throws Exception {
    MiniDFSCluster cluster = null;
    Random random = new Random();
    
    try {
      Configuration conf = new HdfsConfiguration();
      conf.setInt("dfs.datanode.directoryscan.interval", 1); // datanode scans directories
      conf.setInt("dfs.blockreport.intervalMsec", 3 * 1000); // datanode sends block reports
      cluster = new MiniDFSCluster.Builder(conf).build();
      FileSystem fs = cluster.getFileSystem();

      // create two files with one block each
      DFSTestUtil util = new DFSTestUtil("testCorruptFilesCorruptedBlock", 2, 1, 512);
      util.createFiles(fs, "/srcdat10");

      // fetch bad file list from namenode. There should be none.
      final NameNode namenode = cluster.getNameNode();
      Collection<FSNamesystem.CorruptFileBlockInfo> badFiles = namenode
          .listCorruptFileBlocks("/", null);
      assertTrue("Namenode has " + badFiles.size()
          + " corrupt files. Expecting None.", badFiles.size() == 0);

      // Now deliberately corrupt one block
      File data_dir = new File(System.getProperty("test.build.data"),
      "dfs/data/data1/current/finalized");
      assertTrue("data directory does not exist", data_dir.exists());
      File[] blocks = data_dir.listFiles();
      assertTrue("Blocks do not exist in data-dir", (blocks != null) && (blocks.length > 0));
      for (int idx = 0; idx < blocks.length; idx++) {
        if (blocks[idx].getName().startsWith("blk_") &&
            blocks[idx].getName().endsWith(".meta")) {
          //
          // shorten .meta file
          //
          RandomAccessFile file = new RandomAccessFile(blocks[idx], "rw");
          FileChannel channel = file.getChannel();
          long position = channel.size() - 2;
          int length = 2;
          byte[] buffer = new byte[length];
          random.nextBytes(buffer);
          channel.write(ByteBuffer.wrap(buffer), position);
          file.close();
          LOG.info("Deliberately corrupting file " + blocks[idx].getName() +
              " at offset " + position + " length " + length);

          // read all files to trigger detection of corrupted replica
          try {
            util.checkFiles(fs, "/srcdat10");
          } catch (BlockMissingException e) {
            System.out.println("Received BlockMissingException as expected.");
          } catch (IOException e) {
            assertTrue("Corrupted replicas not handled properly. Expecting BlockMissingException " +
                " but received IOException " + e, false);
          }
          break;
        }
      }

      // fetch bad file list from namenode. There should be one file.
      badFiles = namenode.listCorruptFileBlocks("/", null);
      LOG.info("Namenode has bad files. " + badFiles.size());
      assertTrue("Namenode has " + badFiles.size() + " bad files. Expecting 1.",
          badFiles.size() == 1);
      util.cleanup(fs, "/srcdat10");
    } finally {
      if (cluster != null) { cluster.shutdown(); }
    }
  }
  
  // deliberately remove blocks from a file and validate the list-corrupt-file-blocks API
  public void testlistCorruptFileBlocks() throws Exception {
    Configuration conf = new Configuration();
    conf.setLong("dfs.blockreport.intervalMsec", 1000);
    conf.setInt("dfs.datanode.directoryscan.interval", 1); // datanode scans
                                                           // directories
    FileSystem fs = null;

    MiniDFSCluster cluster = null;
    try {
      cluster = new MiniDFSCluster.Builder(conf).build();
      cluster.waitActive();
      fs = cluster.getFileSystem();
      DFSTestUtil util = new DFSTestUtil("testGetCorruptFiles", 3, 1, 1024);
      util.createFiles(fs, "/corruptData");

      final NameNode namenode = cluster.getNameNode();
      Collection<FSNamesystem.CorruptFileBlockInfo> corruptFileBlocks = namenode
          .listCorruptFileBlocks("/corruptData", null);
      int numCorrupt = corruptFileBlocks.size();
      assertTrue(numCorrupt == 0);
      // delete the blocks
      File baseDir = new File(System.getProperty("test.build.data",
          "build/test/data"), "dfs/data");
      for (int i = 0; i < 8; i++) {
        File data_dir = new File(baseDir, "data" + (i + 1)
            + MiniDFSCluster.FINALIZED_DIR_NAME);
        File[] blocks = data_dir.listFiles();
        if (blocks == null)
          continue;
        // assertTrue("Blocks do not exist in data-dir", (blocks != null) &&
        // (blocks.length > 0));
        for (int idx = 0; idx < blocks.length; idx++) {
          if (!blocks[idx].getName().startsWith("blk_")) {
            continue;
          }
          LOG.info("Deliberately removing file " + blocks[idx].getName());
          assertTrue("Cannot remove file.", blocks[idx].delete());
          // break;
        }
      }

      int count = 0;
      corruptFileBlocks = namenode.listCorruptFileBlocks("/corruptData", null);
      numCorrupt = corruptFileBlocks.size();
      while (numCorrupt < 3) {
        Thread.sleep(1000);
        corruptFileBlocks = namenode
            .listCorruptFileBlocks("/corruptData", null);
        numCorrupt = corruptFileBlocks.size();
        count++;
        if (count > 30)
          break;
      }
      // Validate we get all the corrupt files
      LOG.info("Namenode has bad files. " + numCorrupt);
      assertTrue(numCorrupt == 3);
      // test the paging here

      FSNamesystem.CorruptFileBlockInfo[] cfb = corruptFileBlocks
          .toArray(new FSNamesystem.CorruptFileBlockInfo[0]);
      // now get the 2nd and 3rd file that is corrupt
      Collection<FSNamesystem.CorruptFileBlockInfo> nextCorruptFileBlocks = namenode
          .listCorruptFileBlocks("/corruptData", cfb[0].block.getBlockName());
      FSNamesystem.CorruptFileBlockInfo[] ncfb = nextCorruptFileBlocks
          .toArray(new FSNamesystem.CorruptFileBlockInfo[0]);
      numCorrupt = nextCorruptFileBlocks.size();
      assertTrue(numCorrupt == 2);
      assertTrue(ncfb[0].block.getBlockName()
          .equalsIgnoreCase(cfb[1].block.getBlockName()));

      corruptFileBlocks = namenode.listCorruptFileBlocks("/corruptData",
          ncfb[1].block.getBlockName());
      numCorrupt = corruptFileBlocks.size();
      assertTrue(numCorrupt == 0);
      // Do a listing on a dir which doesn't have any corrupt blocks and
      // validate
      util.createFiles(fs, "/goodData");
      corruptFileBlocks = namenode.listCorruptFileBlocks("/goodData", null);
      numCorrupt = corruptFileBlocks.size();
      assertTrue(numCorrupt == 0);
      util.cleanup(fs, "/corruptData");
      util.cleanup(fs, "/goodData");
    } finally {
      if (cluster != null) {
        cluster.shutdown();
      }
    }
  }
  
  /** check if NN.listCorruptFiles() returns the right limit */
  public void testMaxCorruptFiles() throws Exception {
    MiniDFSCluster cluster = null;
    try {
      Configuration conf = new HdfsConfiguration();
      conf.setInt("dfs.datanode.directoryscan.interval", 15); // datanode scans directories
      conf.setInt("dfs.blockreport.intervalMsec", 3 * 1000); // datanode sends block reports
      cluster = new MiniDFSCluster.Builder(conf).build();
      FileSystem fs = cluster.getFileSystem();
      final int maxCorruptFileBlocks = 
        FSNamesystem.DEFAULT_MAX_CORRUPT_FILEBLOCKS_RETURNED;

      // create 110 files with one block each
      DFSTestUtil util = new DFSTestUtil("testMaxCorruptFiles", 
          maxCorruptFileBlocks * 3, 1, 512);
      util.createFiles(fs, "/srcdat2", (short) 1);
      util.waitReplication(fs, "/srcdat2", (short) 1);

      // verify that there are no bad blocks.
      final NameNode namenode = cluster.getNameNode();
      Collection<FSNamesystem.CorruptFileBlockInfo> badFiles = namenode
          .listCorruptFileBlocks("/srcdat2", null);
      assertTrue("Namenode has " + badFiles.size() + " corrupt files. Expecting none.",
          badFiles.size() == 0);

      // Now deliberately blocks from all files
      File baseDir = new File(System.getProperty("test.build.data",
      "build/test/data"),"dfs/data");
      for (int i=0; i<8; i++) {
        File data_dir = new File(baseDir, "data" +(i+1)+ MiniDFSCluster.FINALIZED_DIR_NAME);
        File[] blocks = data_dir.listFiles();
        if (blocks == null)
          continue;

        for (int idx = 0; idx < blocks.length; idx++) {
          if (!blocks[idx].getName().startsWith("blk_")) {
            continue;
          }
          assertTrue("Cannot remove file.", blocks[idx].delete());
        }
      }

      badFiles = namenode.listCorruptFileBlocks("/srcdat2", null);
        
       while (badFiles.size() < maxCorruptFileBlocks) {
        LOG.info("# of corrupt files is: " + badFiles.size());
        Thread.sleep(10000);
        badFiles = namenode.listCorruptFileBlocks("/srcdat2", null);
      }
      badFiles = namenode.listCorruptFileBlocks("/srcdat2", null); 
      LOG.info("Namenode has bad files. " + badFiles.size());
      assertTrue("Namenode has " + badFiles.size() + " bad files. Expecting " + 
          maxCorruptFileBlocks + ".",
          badFiles.size() == maxCorruptFileBlocks);
      util.cleanup(fs, "/srcdat2");
    } finally {
      if (cluster != null) { cluster.shutdown(); }
    }
  }

}

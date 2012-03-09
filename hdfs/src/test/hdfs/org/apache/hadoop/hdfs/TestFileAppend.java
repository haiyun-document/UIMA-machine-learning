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

import java.io.File;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.List;

import junit.framework.TestCase;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileUtil.HardLink;
import org.apache.hadoop.hdfs.protocol.Block;
import org.apache.hadoop.hdfs.protocol.LocatedBlock;
import org.apache.hadoop.hdfs.protocol.LocatedBlocks;
import org.apache.hadoop.hdfs.server.datanode.DataNode;
import org.apache.hadoop.hdfs.server.datanode.FSDataset;
import org.apache.hadoop.hdfs.server.datanode.SimulatedFSDataset;

/**
 * This class tests the building blocks that are needed to
 * support HDFS appends.
 */
public class TestFileAppend extends TestCase {
  boolean simulatedStorage = false;

  private static byte[] fileContents = null;

  //
  // writes to file but does not close it
  //
  private void writeFile(FSDataOutputStream stm) throws IOException {
    byte[] buffer = AppendTestUtil.initBuffer(AppendTestUtil.FILE_SIZE);
    stm.write(buffer);
  }

  //
  // verify that the data written to the full blocks are sane
  // 
  private void checkFile(FileSystem fileSys, Path name, int repl)
    throws IOException {
    boolean done = false;

    // wait till all full blocks are confirmed by the datanodes.
    while (!done) {
      try {
        Thread.sleep(1000);
      } catch (InterruptedException e) {;}
      done = true;
      BlockLocation[] locations = fileSys.getFileBlockLocations(
          fileSys.getFileStatus(name), 0, AppendTestUtil.FILE_SIZE);
      if (locations.length < AppendTestUtil.NUM_BLOCKS) {
        System.out.println("Number of blocks found " + locations.length);
        done = false;
        continue;
      }
      for (int idx = 0; idx < AppendTestUtil.NUM_BLOCKS; idx++) {
        if (locations[idx].getHosts().length < repl) {
          System.out.println("Block index " + idx + " not yet replciated.");
          done = false;
          break;
        }
      }
    }
    byte[] expected = 
        new byte[AppendTestUtil.NUM_BLOCKS * AppendTestUtil.BLOCK_SIZE];
    if (simulatedStorage) {
      for (int i= 0; i < expected.length; i++) {  
        expected[i] = SimulatedFSDataset.DEFAULT_DATABYTE;
      }
    } else {
      System.arraycopy(fileContents, 0, expected, 0, expected.length);
    }
    // do a sanity check. Read the file
    AppendTestUtil.checkFullFile(fileSys, name,
        AppendTestUtil.NUM_BLOCKS * AppendTestUtil.BLOCK_SIZE,
        expected, "Read 1");
  }

  /**
   * Test that copy on write for blocks works correctly
   * @throws IOException an exception might be thrown
   */
  public void testCopyOnWrite() throws IOException {
    Configuration conf = new HdfsConfiguration();
    if (simulatedStorage) {
      conf.setBoolean(SimulatedFSDataset.CONFIG_PROPERTY_SIMULATED, true);
    }
    MiniDFSCluster cluster = new MiniDFSCluster.Builder(conf).build();
    FileSystem fs = cluster.getFileSystem();
    InetSocketAddress addr = new InetSocketAddress("localhost",
                                                   cluster.getNameNodePort());
    DFSClient client = new DFSClient(addr, conf);
    try {

      // create a new file, write to it and close it.
      //
      Path file1 = new Path("/filestatus.dat");
      FSDataOutputStream stm = AppendTestUtil.createFile(fs, file1, 1);
      writeFile(stm);
      stm.close();

      // Get a handle to the datanode
      DataNode[] dn = cluster.listDataNodes();
      assertTrue("There should be only one datanode but found " + dn.length,
                  dn.length == 1);

      LocatedBlocks locations = client.getNamenode().getBlockLocations(
                                  file1.toString(), 0, Long.MAX_VALUE);
      List<LocatedBlock> blocks = locations.getLocatedBlocks();
      FSDataset dataset = (FSDataset) dn[0].data;

      //
      // Create hard links for a few of the blocks
      //
      for (int i = 0; i < blocks.size(); i = i + 2) {
        Block b = blocks.get(i).getBlock();
        File f = dataset.getFile(b);
        File link = new File(f.toString() + ".link");
        System.out.println("Creating hardlink for File " + f + " to " + link);
        HardLink.createHardLink(f, link);
      }

      //
      // Detach all blocks. This should remove hardlinks (if any)
      //
      for (int i = 0; i < blocks.size(); i++) {
        Block b = blocks.get(i).getBlock();
        System.out.println("testCopyOnWrite detaching block " + b);
        assertTrue("Detaching block " + b + " should have returned true",
            dataset.unlinkBlock(b, 1));
      }

      // Since the blocks were already detached earlier, these calls should
      // return false
      //
      for (int i = 0; i < blocks.size(); i++) {
        Block b = blocks.get(i).getBlock();
        System.out.println("testCopyOnWrite detaching block " + b);
        assertTrue("Detaching block " + b + " should have returned false",
            !dataset.unlinkBlock(b, 1));
      }

    } finally {
      fs.close();
      cluster.shutdown();
    }
  }

  /**
   * Test a simple flush on a simple HDFS file.
   * @throws IOException an exception might be thrown
   */
  public void testSimpleFlush() throws IOException {
    Configuration conf = new HdfsConfiguration();
    if (simulatedStorage) {
      conf.setBoolean(SimulatedFSDataset.CONFIG_PROPERTY_SIMULATED, true);
    }
    fileContents = AppendTestUtil.initBuffer(AppendTestUtil.FILE_SIZE);
    MiniDFSCluster cluster = new MiniDFSCluster.Builder(conf).build();
    FileSystem fs = cluster.getFileSystem();
    try {

      // create a new file.
      Path file1 = new Path("/simpleFlush.dat");
      FSDataOutputStream stm = AppendTestUtil.createFile(fs, file1, 1);
      System.out.println("Created file simpleFlush.dat");

      // write to file
      int mid = AppendTestUtil.FILE_SIZE /2;
      stm.write(fileContents, 0, mid);
      stm.hflush();
      System.out.println("Wrote and Flushed first part of file.");

      // write the remainder of the file
      stm.write(fileContents, mid, AppendTestUtil.FILE_SIZE - mid);
      System.out.println("Written second part of file");
      stm.hflush();
      stm.hflush();
      System.out.println("Wrote and Flushed second part of file.");

      // verify that full blocks are sane
      checkFile(fs, file1, 1);

      stm.close();
      System.out.println("Closed file.");

      // verify that entire file is good
      AppendTestUtil.checkFullFile(fs, file1, AppendTestUtil.FILE_SIZE,
          fileContents, "Read 2");

    } catch (IOException e) {
      System.out.println("Exception :" + e);
      throw e; 
    } catch (Throwable e) {
      System.out.println("Throwable :" + e);
      e.printStackTrace();
      throw new IOException("Throwable : " + e);
    } finally {
      fs.close();
      cluster.shutdown();
    }
  }

  /**
   * Test that file data can be flushed.
   * @throws IOException an exception might be thrown
   */
  public void testComplexFlush() throws IOException {
    Configuration conf = new HdfsConfiguration();
    if (simulatedStorage) {
      conf.setBoolean(SimulatedFSDataset.CONFIG_PROPERTY_SIMULATED, true);
    }
    fileContents = AppendTestUtil.initBuffer(AppendTestUtil.FILE_SIZE);
    MiniDFSCluster cluster = new MiniDFSCluster.Builder(conf).build();
    FileSystem fs = cluster.getFileSystem();
    try {

      // create a new file.
      Path file1 = new Path("/complexFlush.dat");
      FSDataOutputStream stm = AppendTestUtil.createFile(fs, file1, 1);
      System.out.println("Created file complexFlush.dat");

      int start = 0;
      for (start = 0; (start + 29) < AppendTestUtil.FILE_SIZE; ) {
        stm.write(fileContents, start, 29);
        stm.hflush();
        start += 29;
      }
      stm.write(fileContents, start, AppendTestUtil.FILE_SIZE -start);

      // verify that full blocks are sane
      checkFile(fs, file1, 1);
      stm.close();

      // verify that entire file is good
      AppendTestUtil.checkFullFile(fs, file1, AppendTestUtil.FILE_SIZE,
          fileContents, "Read 2");
    } catch (IOException e) {
      System.out.println("Exception :" + e);
      throw e; 
    } catch (Throwable e) {
      System.out.println("Throwable :" + e);
      e.printStackTrace();
      throw new IOException("Throwable : " + e);
    } finally {
      fs.close();
      cluster.shutdown();
    }
  }
}

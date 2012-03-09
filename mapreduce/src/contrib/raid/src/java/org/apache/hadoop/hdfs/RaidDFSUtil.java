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
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.LinkedList;
import java.util.List;
import java.util.HashSet;
import java.util.Set;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.hdfs.protocol.LocatedBlock;
import org.apache.hadoop.hdfs.protocol.LocatedBlocks;
import org.apache.hadoop.hdfs.tools.DFSck;
import org.apache.hadoop.util.ToolRunner;

public abstract class RaidDFSUtil {
  /**
   * Returns the corrupt blocks in a file.
   */
  public static List<LocatedBlock> corruptBlocksInFile(
    DistributedFileSystem dfs, String path, long offset, long length)
  throws IOException {
    List<LocatedBlock> corrupt = new LinkedList<LocatedBlock>();
    LocatedBlocks locatedBlocks =
      getBlockLocations(dfs, path, offset, length);
    for (LocatedBlock b: locatedBlocks.getLocatedBlocks()) {
      if (b.isCorrupt() ||
         (b.getLocations().length == 0 && b.getBlockSize() > 0)) {
        corrupt.add(b);
      }
    }
    return corrupt;
  }

  public static LocatedBlocks getBlockLocations(
    DistributedFileSystem dfs, String path, long offset, long length)
    throws IOException {
    return dfs.getClient().namenode.getBlockLocations(path, offset, length);
  }

  public static String[] getCorruptFiles(Configuration conf)
    throws IOException {
    ByteArrayOutputStream baseOut = new ByteArrayOutputStream();
    PrintStream out = new PrintStream(baseOut, true);
    DFSck fsck = new DFSck(conf, out);
    String[] args = new String[]{"-list-corruptfileblocks"};
    try {
      ToolRunner.run(fsck, args);
    } catch (Exception e) {
      throw new IOException("DFSck.run exception ", e);
    }
    byte[] output = baseOut.toByteArray();
    BufferedReader in = new BufferedReader(new InputStreamReader(
      new ByteArrayInputStream(output)));
    String line;
    Set<String> corruptFiles = new HashSet<String>();
    while ((line = in.readLine()) != null) {
      // The interesting lines are of the form: blkid<tab>path
      int separatorPos = line.indexOf('\t');
      if (separatorPos != -1) {
        corruptFiles.add(line.substring(separatorPos + 1));
      }
    }
    return corruptFiles.toArray(new String[corruptFiles.size()]);
  }
}

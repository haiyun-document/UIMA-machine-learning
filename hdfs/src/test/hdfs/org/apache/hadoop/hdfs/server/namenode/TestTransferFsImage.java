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

import static org.junit.Assert.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hdfs.HdfsConfiguration;
import org.apache.hadoop.hdfs.MiniDFSCluster;
import org.apache.hadoop.util.StringUtils;
import org.junit.Test;

public class TestTransferFsImage {

  /**
   * Regression test for HDFS-1997. Test that, if an exception
   * occurs on the client side, it is properly reported as such
   */
  @Test
  public void testClientSideException() throws IOException {

    Configuration conf = new HdfsConfiguration();
    MiniDFSCluster cluster = new MiniDFSCluster.Builder(conf)
      .numDataNodes(0).build();
    try {
      
      String fsName = NameNode.getHostPortString(
          cluster.getNameNode().getHttpAddress());
      String id = "getimage=1";

      File[] localPath = new File[] {
         new File("/xxxxx-does-not-exist/blah") 
      };
    
      TransferFsImage.getFileClient(fsName, id, localPath);
      fail("Didn't get an exception!");
    } catch (IOException ioe) {
      assertTrue("Expected FNFE, got: " + StringUtils.stringifyException(ioe),
          ioe instanceof FileNotFoundException);
    } finally {
      cluster.shutdown();      
    }
  }
}

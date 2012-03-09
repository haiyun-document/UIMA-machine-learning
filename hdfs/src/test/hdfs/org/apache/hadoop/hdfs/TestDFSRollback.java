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

import static org.apache.hadoop.hdfs.server.common.HdfsConstants.NodeType.DATA_NODE;
import static org.apache.hadoop.hdfs.server.common.HdfsConstants.NodeType.NAME_NODE;

import java.io.File;
import java.io.IOException;

import junit.framework.TestCase;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.hdfs.server.common.StorageInfo;
import org.apache.hadoop.hdfs.server.common.HdfsConstants.NodeType;
import org.apache.hadoop.hdfs.server.common.HdfsConstants.StartupOption;
import org.apache.hadoop.util.StringUtils;

/**
* This test ensures the appropriate response (successful or failure) from
* the system when the system is rolled back under various storage state and
* version conditions.
*/
public class TestDFSRollback extends TestCase {
 
  private static final Log LOG = LogFactory.getLog(
                                                   "org.apache.hadoop.hdfs.TestDFSRollback");
  private Configuration conf;
  private int testCounter = 0;
  private MiniDFSCluster cluster = null;
  
  /**
   * Writes an INFO log message containing the parameters.
   */
  void log(String label, int numDirs) {
    LOG.info("============================================================");
    LOG.info("***TEST " + (testCounter++) + "*** " 
             + label + ":"
             + " numDirs="+numDirs);
  }
  
  /**
   * Verify that the new current directory is the old previous.  
   * It is assumed that the server has recovered and rolled back.
   */
  void checkResult(NodeType nodeType, String[] baseDirs) throws IOException {
    switch (nodeType) {
    case NAME_NODE:
      for (int i = 0; i < baseDirs.length; i++) {
        assertTrue(new File(baseDirs[i],"current").isDirectory());
        assertTrue(new File(baseDirs[i],"current/VERSION").isFile());
        assertTrue(new File(baseDirs[i],"current/edits").isFile());
        assertTrue(new File(baseDirs[i],"current/fsimage").isFile());
        assertTrue(new File(baseDirs[i],"current/fstime").isFile());
      }
      break;
    case DATA_NODE:
      for (int i = 0; i < baseDirs.length; i++) {
        assertEquals(
                     UpgradeUtilities.checksumContents(
                                                       nodeType, new File(baseDirs[i],"current")),
                     UpgradeUtilities.checksumMasterContents(nodeType));
      }
      break;
    }
    for (int i = 0; i < baseDirs.length; i++) {
      assertFalse(new File(baseDirs[i],"previous").isDirectory());
    }
  }
 
  /**
   * Attempts to start a NameNode with the given operation.  Starting
   * the NameNode should throw an exception.
   */
  void startNameNodeShouldFail(StartupOption operation, String searchString) {
    try {
      cluster = new MiniDFSCluster.Builder(conf).numDataNodes(0)
                                                .startupOption(operation)
                                                .format(false)
                                                .manageDataDfsDirs(false)
                                                .manageNameDfsDirs(false)
                                                .build(); // should fail
      throw new AssertionError("NameNode should have failed to start");
    } catch (Exception expected) {
      if (!expected.getMessage().contains(searchString)) {
        fail("Expected substring '" + searchString + "' in exception " +
            "but got: " + StringUtils.stringifyException(expected));
      }
      // expected
    }
  }
  
  /**
   * Attempts to start a DataNode with the given operation.  Starting
   * the DataNode should throw an exception.
   */
  void startDataNodeShouldFail(StartupOption operation) {
    try {
      cluster.startDataNodes(conf, 1, false, operation, null); // should fail
      throw new AssertionError("DataNode should have failed to start");
    } catch (Exception expected) {
      // expected
      assertFalse(cluster.isDataNodeUp());
    }
  }
 
  /**
   * This test attempts to rollback the NameNode and DataNode under
   * a number of valid and invalid conditions.
   */
  public void testRollback() throws Exception {
    File[] baseDirs;
    UpgradeUtilities.initialize();
    
    for (int numDirs = 1; numDirs <= 2; numDirs++) {
      conf = new HdfsConfiguration();
      conf.setInt("dfs.datanode.scan.period.hours", -1);      
      conf = UpgradeUtilities.initializeStorageStateConf(numDirs, conf);
      String[] nameNodeDirs = conf.getStrings(DFSConfigKeys.DFS_NAMENODE_NAME_DIR_KEY);
      String[] dataNodeDirs = conf.getStrings(DFSConfigKeys.DFS_DATANODE_DATA_DIR_KEY);
      
      log("Normal NameNode rollback", numDirs);
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "current");
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "previous");
      cluster = new MiniDFSCluster.Builder(conf).numDataNodes(0)
                                                .format(false)
                                                .manageDataDfsDirs(false)
                                                .manageNameDfsDirs(false)
                                                .startupOption(StartupOption.ROLLBACK)
                                                .build();
      checkResult(NAME_NODE, nameNodeDirs);
      cluster.shutdown();
      UpgradeUtilities.createEmptyDirs(nameNodeDirs);
      
      log("Normal DataNode rollback", numDirs);
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "current");
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "previous");
      cluster = new MiniDFSCluster.Builder(conf).numDataNodes(0)
                                                .format(false)
                                                .manageDataDfsDirs(false)
                                                .manageNameDfsDirs(false)
                                                .startupOption(StartupOption.ROLLBACK)
                                                .build();
      UpgradeUtilities.createStorageDirs(DATA_NODE, dataNodeDirs, "current");
      UpgradeUtilities.createStorageDirs(DATA_NODE, dataNodeDirs, "previous");
      cluster.startDataNodes(conf, 1, false, StartupOption.ROLLBACK, null);
      checkResult(DATA_NODE, dataNodeDirs);
      cluster.shutdown();
      UpgradeUtilities.createEmptyDirs(nameNodeDirs);
      UpgradeUtilities.createEmptyDirs(dataNodeDirs);

      log("NameNode rollback without existing previous dir", numDirs);
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "current");
      startNameNodeShouldFail(StartupOption.ROLLBACK,
          "None of the storage directories contain previous fs state");
      UpgradeUtilities.createEmptyDirs(nameNodeDirs);
      
      log("DataNode rollback without existing previous dir", numDirs);
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "current");
      cluster = new MiniDFSCluster.Builder(conf).numDataNodes(0)
                                                .format(false)
                                                .manageDataDfsDirs(false)
                                                .manageNameDfsDirs(false)
                                                .startupOption(StartupOption.UPGRADE)
                                                .build();
      UpgradeUtilities.createStorageDirs(DATA_NODE, dataNodeDirs, "current");
      cluster.startDataNodes(conf, 1, false, StartupOption.ROLLBACK, null);
      cluster.shutdown();
      UpgradeUtilities.createEmptyDirs(nameNodeDirs);
      UpgradeUtilities.createEmptyDirs(dataNodeDirs);

      log("DataNode rollback with future stored layout version in previous", numDirs);
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "current");
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "previous");
      cluster = new MiniDFSCluster.Builder(conf).numDataNodes(0)
                                                .format(false)
                                                .manageDataDfsDirs(false)
                                                .manageNameDfsDirs(false)
                                                .startupOption(StartupOption.ROLLBACK)
                                                .build();
      UpgradeUtilities.createStorageDirs(DATA_NODE, dataNodeDirs, "current");
      baseDirs = UpgradeUtilities.createStorageDirs(DATA_NODE, dataNodeDirs, "previous");
      UpgradeUtilities.createVersionFile(DATA_NODE, baseDirs,
                                         new StorageInfo(Integer.MIN_VALUE,
                                                         UpgradeUtilities.getCurrentNamespaceID(cluster),
                                                         UpgradeUtilities.getCurrentFsscTime(cluster)));
      startDataNodeShouldFail(StartupOption.ROLLBACK);
      cluster.shutdown();
      UpgradeUtilities.createEmptyDirs(nameNodeDirs);
      UpgradeUtilities.createEmptyDirs(dataNodeDirs);
      
      log("DataNode rollback with newer fsscTime in previous", numDirs);
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "current");
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "previous");
      cluster = new MiniDFSCluster.Builder(conf).numDataNodes(0)
                                                .format(false)
                                                .manageDataDfsDirs(false)
                                                .manageNameDfsDirs(false)
                                                .startupOption(StartupOption.ROLLBACK)
                                                .build();
      UpgradeUtilities.createStorageDirs(DATA_NODE, dataNodeDirs, "current");
      baseDirs = UpgradeUtilities.createStorageDirs(DATA_NODE, dataNodeDirs, "previous");
      UpgradeUtilities.createVersionFile(DATA_NODE, baseDirs,
                                         new StorageInfo(UpgradeUtilities.getCurrentLayoutVersion(),
                                                         UpgradeUtilities.getCurrentNamespaceID(cluster),
                                                         Long.MAX_VALUE));
      startDataNodeShouldFail(StartupOption.ROLLBACK);
      cluster.shutdown();
      UpgradeUtilities.createEmptyDirs(nameNodeDirs);
      UpgradeUtilities.createEmptyDirs(dataNodeDirs);

      log("NameNode rollback with no edits file", numDirs);
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "current");
      baseDirs = UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "previous");
      for (File f : baseDirs) { 
        FileUtil.fullyDelete(new File(f,"edits"));
      }
      startNameNodeShouldFail(StartupOption.ROLLBACK,
          "Edits file is not found");
      UpgradeUtilities.createEmptyDirs(nameNodeDirs);
      
      log("NameNode rollback with no image file", numDirs);
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "current");
      baseDirs = UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "previous");
      for (File f : baseDirs) { 
        FileUtil.fullyDelete(new File(f,"fsimage")); 
      }
      startNameNodeShouldFail(StartupOption.ROLLBACK,
          "Image file is not found");
      UpgradeUtilities.createEmptyDirs(nameNodeDirs);
      
      log("NameNode rollback with corrupt version file", numDirs);
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "current");
      baseDirs = UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "previous");
      for (File f : baseDirs) { 
        UpgradeUtilities.corruptFile(new File(f,"VERSION")); 
      }
      startNameNodeShouldFail(StartupOption.ROLLBACK,
          "file VERSION is invalid");
      UpgradeUtilities.createEmptyDirs(nameNodeDirs);
      
      log("NameNode rollback with old layout version in previous", numDirs);
      UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "current");
      baseDirs = UpgradeUtilities.createStorageDirs(NAME_NODE, nameNodeDirs, "previous");
      UpgradeUtilities.createVersionFile(NAME_NODE, baseDirs,
                                         new StorageInfo(1,
                                                         UpgradeUtilities.getCurrentNamespaceID(null),
                                                         UpgradeUtilities.getCurrentFsscTime(null)));
      startNameNodeShouldFail(StartupOption.ROLLBACK,
          "Cannot rollback to storage version 1 using this version");
      UpgradeUtilities.createEmptyDirs(nameNodeDirs);
    } // end numDir loop
  }
 
  protected void tearDown() throws Exception {
    LOG.info("Shutting down MiniDFSCluster");
    if (cluster != null) cluster.shutdown();
  }
  
  public static void main(String[] args) throws Exception {
    new TestDFSRollback().testRollback();
  }
  
}



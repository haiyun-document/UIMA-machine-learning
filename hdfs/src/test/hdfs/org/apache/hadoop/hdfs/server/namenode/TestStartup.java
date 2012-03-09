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

import static org.apache.hadoop.hdfs.server.common.HdfsConstants.StartupOption.IMPORT;
import static org.apache.hadoop.hdfs.server.common.Util.fileAsURI;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.URI;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.Random;

import junit.framework.TestCase;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.fs.permission.PermissionStatus;

import org.apache.hadoop.hdfs.DFSConfigKeys;
import org.apache.hadoop.hdfs.HdfsConfiguration;
import org.apache.hadoop.hdfs.MiniDFSCluster;
import org.apache.hadoop.hdfs.protocol.FSConstants.SafeModeAction;
import org.apache.hadoop.hdfs.server.common.Storage.StorageDirectory;
import org.apache.hadoop.hdfs.server.namenode.FSImage.NameNodeDirType;
import org.apache.hadoop.hdfs.server.namenode.FSImage.NameNodeFile;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.MD5Hash;
import org.apache.hadoop.util.StringUtils;

/**
 * Startup and checkpoint tests
 * 
 */
public class TestStartup extends TestCase {
  public static final String NAME_NODE_HOST = "localhost:";
  public static final String WILDCARD_HTTP_HOST = "0.0.0.0:";
  private static final Log LOG =
    LogFactory.getLog(TestStartup.class.getName());
  private Configuration config;
  private File hdfsDir=null;
  static final long seed = 0xAAAAEEFL;
  static final int blockSize = 4096;
  static final int fileSize = 8192;
  private long editsLength=0, fsimageLength=0;


  private void writeFile(FileSystem fileSys, Path name, int repl)
  throws IOException {
    FSDataOutputStream stm = fileSys.create(name, true,
        fileSys.getConf().getInt("io.file.buffer.size", 4096),
        (short)repl, (long)blockSize);
    byte[] buffer = new byte[fileSize];
    Random rand = new Random(seed);
    rand.nextBytes(buffer);
    stm.write(buffer);
    stm.close();
  }


  protected void setUp() throws Exception {
    config = new HdfsConfiguration();
    hdfsDir = new File(MiniDFSCluster.getBaseDirectory());

    if ( hdfsDir.exists() && !FileUtil.fullyDelete(hdfsDir) ) {
      throw new IOException("Could not delete hdfs directory '" + hdfsDir + "'");
    }
    LOG.info("--hdfsdir is " + hdfsDir.getAbsolutePath());
    config.set(DFSConfigKeys.DFS_NAMENODE_NAME_DIR_KEY,
        fileAsURI(new File(hdfsDir, "name")).toString());
    config.set(DFSConfigKeys.DFS_DATANODE_DATA_DIR_KEY,
        new File(hdfsDir, "data").getPath());
    config.set(DFSConfigKeys.DFS_NAMENODE_CHECKPOINT_DIR_KEY,
        fileAsURI(new File(hdfsDir, "secondary")).toString());
    config.set(DFSConfigKeys.DFS_NAMENODE_SECONDARY_HTTP_ADDRESS_KEY,
	       WILDCARD_HTTP_HOST + "0");
    
    FileSystem.setDefaultUri(config, "hdfs://"+NAME_NODE_HOST + "0");
  }

  /**
   * clean up
   */
  public void tearDown() throws Exception {
    if ( hdfsDir.exists() && !FileUtil.fullyDelete(hdfsDir) ) {
      throw new IOException("Could not delete hdfs directory in tearDown '" + hdfsDir + "'");
    }	
  }

   /**
   * start MiniDFScluster, create a file (to create edits) and do a checkpoint  
   * @throws IOException
   */
  @SuppressWarnings("deprecation")
  public void createCheckPoint() throws IOException {
    LOG.info("--starting mini cluster");
    // manage dirs parameter set to false 
    MiniDFSCluster cluster = null;
    SecondaryNameNode sn = null;
    
    try {
      cluster = new MiniDFSCluster.Builder(config)
                                  .manageDataDfsDirs(false)
                                  .manageNameDfsDirs(false).build();
      cluster.waitActive();

      LOG.info("--starting Secondary Node");

      // start secondary node
      sn = new SecondaryNameNode(config);
      assertNotNull(sn);

      // create a file
      FileSystem fileSys = cluster.getFileSystem();
      Path file1 = new Path("t1");
      this.writeFile(fileSys, file1, 1);

      LOG.info("--doing checkpoint");
      sn.doCheckpoint();  // this shouldn't fail
      LOG.info("--done checkpoint");
    } catch (IOException e) {
      fail(StringUtils.stringifyException(e));
      System.err.println("checkpoint failed");
      throw e;
    }  finally {
      if(sn!=null)
        sn.shutdown();
      if(cluster!=null) 
        cluster.shutdown();
      LOG.info("--file t1 created, cluster shutdown");
    }
  }

  /*
   * corrupt files by removing and recreating the directory
   */
  private void corruptNameNodeFiles() throws IOException {
    // now corrupt/delete the directrory
    List<URI> nameDirs = (List<URI>)FSNamesystem.getNamespaceDirs(config);
    List<URI> nameEditsDirs = (List<URI>)FSNamesystem.getNamespaceEditsDirs(config);

    // get name dir and its length, then delete and recreate the directory
    File dir = new File(nameDirs.get(0).getPath()); // has only one
    this.fsimageLength = new File(new File(dir, "current"), 
        NameNodeFile.IMAGE.getName()).length();

    if(dir.exists() && !(FileUtil.fullyDelete(dir)))
      throw new IOException("Cannot remove directory: " + dir);

    LOG.info("--removed dir "+dir + ";len was ="+ this.fsimageLength);

    if (!dir.mkdirs())
      throw new IOException("Cannot create directory " + dir);

    dir = new File( nameEditsDirs.get(0).getPath()); //has only one

    this.editsLength = new File(new File(dir, "current"), 
        NameNodeFile.EDITS.getName()).length();

    if(dir.exists() && !(FileUtil.fullyDelete(dir)))
      throw new IOException("Cannot remove directory: " + dir);
    if (!dir.mkdirs())
      throw new IOException("Cannot create directory " + dir);

    LOG.info("--removed dir and recreated "+dir + ";len was ="+ this.editsLength);


  }

  /**
   * start with -importCheckpoint option and verify that the files are in separate directories and of the right length
   * @throws IOException
   */
  private void checkNameNodeFiles() throws IOException{

    // start namenode with import option
    LOG.info("-- about to start DFS cluster");
    MiniDFSCluster cluster = null;
    try {
      cluster = new MiniDFSCluster.Builder(config)
                                  .format(false)
                                  .manageDataDfsDirs(false)
                                  .manageNameDfsDirs(false)
                                  .startupOption(IMPORT).build();
      cluster.waitActive();
      LOG.info("--NN started with checkpoint option");
      NameNode nn = cluster.getNameNode();
      assertNotNull(nn);	
      // Verify that image file sizes did not change.
      FSImage image = nn.getFSImage();
      verifyDifferentDirs(image, this.fsimageLength, this.editsLength);
    } finally {
      if(cluster != null)
        cluster.shutdown();
    }
  }

  /**
   * verify that edits log and fsimage are in different directories and of a correct size
   */
  private void verifyDifferentDirs(FSImage img, long expectedImgSize, long expectedEditsSize) {
    StorageDirectory sd =null;
    for (Iterator<StorageDirectory> it = img.dirIterator(); it.hasNext();) {
      sd = it.next();

      if(sd.getStorageDirType().isOfType(NameNodeDirType.IMAGE)) {
        File imf = FSImage.getImageFile(sd, NameNodeFile.IMAGE);
        LOG.info("--image file " + imf.getAbsolutePath() + "; len = " + imf.length() + "; expected = " + expectedImgSize);
        assertEquals(expectedImgSize, imf.length());	
      } else if(sd.getStorageDirType().isOfType(NameNodeDirType.EDITS)) {
        File edf = FSImage.getImageFile(sd, NameNodeFile.EDITS);
        LOG.info("-- edits file " + edf.getAbsolutePath() + "; len = " + edf.length()  + "; expected = " + expectedEditsSize);
        assertEquals(expectedEditsSize, edf.length());	
      } else {
        fail("Image/Edits directories are not different");
      }
    }

  }
  /**
   * secnn-6
   * checkpoint for edits and image is the same directory
   * @throws IOException
   */
  public void testChkpointStartup2() throws IOException{
    LOG.info("--starting checkpointStartup2 - same directory for checkpoint");
    // different name dirs
    config.set(DFSConfigKeys.DFS_NAMENODE_NAME_DIR_KEY,
        fileAsURI(new File(hdfsDir, "name")).toString());
    config.set(DFSConfigKeys.DFS_NAMENODE_EDITS_DIR_KEY,
        fileAsURI(new File(hdfsDir, "edits")).toString());
    // same checkpoint dirs
    config.set(DFSConfigKeys.DFS_NAMENODE_CHECKPOINT_EDITS_DIR_KEY,
        fileAsURI(new File(hdfsDir, "chkpt")).toString());
    config.set(DFSConfigKeys.DFS_NAMENODE_CHECKPOINT_DIR_KEY,
        fileAsURI(new File(hdfsDir, "chkpt")).toString());

    createCheckPoint();

    corruptNameNodeFiles();
    checkNameNodeFiles();

  }

  /**
   * seccn-8
   * checkpoint for edits and image are different directories 
   * @throws IOException
   */
  public void testChkpointStartup1() throws IOException{
    //setUpConfig();
    LOG.info("--starting testStartup Recovery");
    // different name dirs
    config.set(DFSConfigKeys.DFS_NAMENODE_NAME_DIR_KEY,
        fileAsURI(new File(hdfsDir, "name")).toString());
    config.set(DFSConfigKeys.DFS_NAMENODE_EDITS_DIR_KEY,
        fileAsURI(new File(hdfsDir, "edits")).toString());
    // same checkpoint dirs
    config.set(DFSConfigKeys.DFS_NAMENODE_CHECKPOINT_EDITS_DIR_KEY,
        fileAsURI(new File(hdfsDir, "chkpt_edits")).toString());
    config.set(DFSConfigKeys.DFS_NAMENODE_CHECKPOINT_DIR_KEY,
        fileAsURI(new File(hdfsDir, "chkpt")).toString());

    createCheckPoint();
    corruptNameNodeFiles();
    checkNameNodeFiles();
  }

  /**
   * secnn-7
   * secondary node copies fsimage and edits into correct separate directories.
   * @throws IOException
   */
  @SuppressWarnings("deprecation")
  public void testSNNStartup() throws IOException{
    //setUpConfig();
    LOG.info("--starting SecondNN startup test");
    // different name dirs
    config.set(DFSConfigKeys.DFS_NAMENODE_NAME_DIR_KEY,
        fileAsURI(new File(hdfsDir, "name")).toString());
    config.set(DFSConfigKeys.DFS_NAMENODE_EDITS_DIR_KEY,
        fileAsURI(new File(hdfsDir, "name")).toString());
    // same checkpoint dirs
    config.set(DFSConfigKeys.DFS_NAMENODE_CHECKPOINT_EDITS_DIR_KEY,
        fileAsURI(new File(hdfsDir, "chkpt_edits")).toString());
    config.set(DFSConfigKeys.DFS_NAMENODE_CHECKPOINT_DIR_KEY,
        fileAsURI(new File(hdfsDir, "chkpt")).toString());

    LOG.info("--starting NN ");
    MiniDFSCluster cluster = null;
    SecondaryNameNode sn = null;
    NameNode nn = null;
    try {
      cluster = new MiniDFSCluster.Builder(config).manageDataDfsDirs(false)
                                                  .manageNameDfsDirs(false)
                                                  .build();
      cluster.waitActive();
      nn = cluster.getNameNode();
      assertNotNull(nn);

      // start secondary node
      LOG.info("--starting SecondNN");
      sn = new SecondaryNameNode(config);
      assertNotNull(sn);

      LOG.info("--doing checkpoint");
      sn.doCheckpoint();  // this shouldn't fail
      LOG.info("--done checkpoint");



      // now verify that image and edits are created in the different directories
      FSImage image = nn.getFSImage();
      StorageDirectory sd = image.getStorageDir(0); //only one
      assertEquals(sd.getStorageDirType(), NameNodeDirType.IMAGE_AND_EDITS);
      File imf = FSImage.getImageFile(sd, NameNodeFile.IMAGE);
      File edf = FSImage.getImageFile(sd, NameNodeFile.EDITS);
      LOG.info("--image file " + imf.getAbsolutePath() + "; len = " + imf.length());
      LOG.info("--edits file " + edf.getAbsolutePath() + "; len = " + edf.length());

      FSImage chkpImage = sn.getFSImage();
      verifyDifferentDirs(chkpImage, imf.length(), edf.length());

    } catch (IOException e) {
      fail(StringUtils.stringifyException(e));
      System.err.println("checkpoint failed");
      throw e;
    } finally {
      if(sn!=null)
        sn.shutdown();
      if(cluster!=null)
        cluster.shutdown();
    }
  }
  
  public void testCompression() throws IOException {
    LOG.info("Test compressing image.");
    Configuration conf = new Configuration();
    FileSystem.setDefaultUri(conf, "hdfs://localhost:0");
    conf.set("dfs.http.address", "127.0.0.1:0");
    File base_dir = new File(System.getProperty(
        "test.build.data", "build/test/data"), "dfs/");
    conf.set("dfs.name.dir", new File(base_dir, "name").getPath());
    conf.setBoolean("dfs.permissions", false);

    NameNode.format(conf);

    // create an uncompressed image
    LOG.info("Create an uncompressed fsimage");
    NameNode namenode = new NameNode(conf);
    namenode.getNamesystem().mkdirs("/test",
        new PermissionStatus("hairong", null, FsPermission.getDefault()), true);
    assertTrue(namenode.getFileInfo("/test").isDir());
    namenode.setSafeMode(SafeModeAction.SAFEMODE_ENTER);
    namenode.saveNamespace();
    namenode.stop();
    namenode.join();

    // compress image using default codec
    LOG.info("Read an uncomressed image and store it compressed using default codec.");
    conf.setBoolean(DFSConfigKeys.DFS_IMAGE_COMPRESS_KEY, true);
    checkNameSpace(conf);

    // read image compressed using the default and compress it using Gzip codec
    LOG.info("Read a compressed image and store it using a different codec.");
    conf.set(DFSConfigKeys.DFS_IMAGE_COMPRESSION_CODEC_KEY,
        "org.apache.hadoop.io.compress.GzipCodec");
    checkNameSpace(conf);

    // read an image compressed in Gzip and store it uncompressed
    LOG.info("Read an compressed iamge and store it as uncompressed.");
    conf.setBoolean(DFSConfigKeys.DFS_IMAGE_COMPRESS_KEY, false);
    checkNameSpace(conf);

    // read an uncomrpessed image and store it uncompressed
    LOG.info("Read an uncompressed image and store it as uncompressed.");
    checkNameSpace(conf);
  }

  private void checkNameSpace(Configuration conf) throws IOException {
    NameNode namenode = new NameNode(conf);
    assertTrue(namenode.getFileInfo("/test").isDir());
    namenode.setSafeMode(SafeModeAction.SAFEMODE_ENTER);
    namenode.saveNamespace();
    namenode.stop();
    namenode.join();
  }
  
  public void testImageChecksum() throws Exception {
    LOG.info("Test uncompressed image checksum");
    testImageChecksum(false);
    LOG.info("Test compressed image checksum");
    testImageChecksum(true);
  }

  private void testImageChecksum(boolean compress) throws Exception {
    Configuration conf = new Configuration();
    FileSystem.setDefaultUri(conf, "hdfs://localhost:0");
    conf.set("dfs.http.address", "127.0.0.1:0");
    File base_dir = new File(
        System.getProperty("test.build.data", "build/test/data"), "dfs/");
    conf.set("dfs.name.dir", new File(base_dir, "name").getPath());
    conf.setBoolean("dfs.permissions", false);
    if (compress) {
      conf.setBoolean(DFSConfigKeys.DFS_IMAGE_COMPRESSION_CODEC_KEY, true);
    }

    NameNode.format(conf);

    // create an image
    LOG.info("Create an fsimage");
    NameNode namenode = new NameNode(conf);
    namenode.getNamesystem().mkdirs("/test",
        new PermissionStatus("hairong", null, FsPermission.getDefault()), true);
    assertTrue(namenode.getFileInfo("/test").isDir());
    namenode.setSafeMode(SafeModeAction.SAFEMODE_ENTER);
    namenode.saveNamespace();

    FSImage image = namenode.getFSImage();
    image.loadFSImage();

    File versionFile = image.getStorageDir(0).getVersionFile();

    RandomAccessFile file = new RandomAccessFile(versionFile, "rws");
    FileInputStream in = null;
    FileOutputStream out = null;
    try {
      // read the property from version file
      in = new FileInputStream(file.getFD());
      file.seek(0);
      Properties props = new Properties();
      props.load(in);

      // get the MD5 property and change it
      String sMd5 = props.getProperty(FSImage.MESSAGE_DIGEST_PROPERTY);
      MD5Hash md5 = new MD5Hash(sMd5);
      byte[] bytes = md5.getDigest();
      bytes[0] += 1;
      md5 = new MD5Hash(bytes);
      props.setProperty(FSImage.MESSAGE_DIGEST_PROPERTY, md5.toString());

      // write the properties back to version file
      file.seek(0);
      out = new FileOutputStream(file.getFD());
      props.store(out, null);
      out.flush();
      file.setLength(out.getChannel().position());

      // now load the image again
      image.loadFSImage();

      fail("Expect to get a checksumerror");
    } catch(IOException e) {
        assertTrue(e.getMessage().contains("is corrupt"));
    } finally {
      IOUtils.closeStream(in);
      IOUtils.closeStream(out);
      namenode.stop();
      namenode.join();
    }
  }
}

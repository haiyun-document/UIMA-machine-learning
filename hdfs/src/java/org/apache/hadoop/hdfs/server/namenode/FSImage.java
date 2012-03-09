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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.URI;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hdfs.protocol.FSConstants;
import org.apache.hadoop.hdfs.protocol.LayoutVersion;
import org.apache.hadoop.hdfs.protocol.LayoutVersion.Feature;
import org.apache.hadoop.hdfs.server.common.InconsistentFSStateException;
import org.apache.hadoop.hdfs.server.common.Storage;
import org.apache.hadoop.hdfs.server.common.StorageInfo;
import org.apache.hadoop.hdfs.server.common.UpgradeManager;
import org.apache.hadoop.hdfs.server.common.Util;
import static org.apache.hadoop.hdfs.server.common.Util.now;
import org.apache.hadoop.hdfs.server.common.HdfsConstants.NamenodeRole;
import org.apache.hadoop.hdfs.server.common.HdfsConstants.NodeType;
import org.apache.hadoop.hdfs.server.common.HdfsConstants.StartupOption;
import org.apache.hadoop.hdfs.server.namenode.JournalStream.JournalType;
import org.apache.hadoop.hdfs.server.protocol.CheckpointCommand;
import org.apache.hadoop.hdfs.server.protocol.NamenodeCommand;
import org.apache.hadoop.hdfs.server.protocol.NamenodeProtocol;
import org.apache.hadoop.hdfs.server.protocol.NamenodeRegistration;
import org.apache.hadoop.io.MD5Hash;
import org.apache.hadoop.hdfs.DFSConfigKeys;

/**
 * FSImage handles checkpointing and logging of the namespace edits.
 * 
 */
@InterfaceAudience.Private
@InterfaceStability.Evolving
public class FSImage extends Storage {

  private static final SimpleDateFormat DATE_FORM =
    new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

  static final String MESSAGE_DIGEST_PROPERTY = "imageMD5Digest";
  //
  // The filenames used for storing the images
  //
  enum NameNodeFile {
    IMAGE     ("fsimage"),
    TIME      ("fstime"),
    EDITS     ("edits"),
    IMAGE_NEW ("fsimage.ckpt"),
    EDITS_NEW ("edits.new");
    
    private String fileName = null;
    private NameNodeFile(String name) {this.fileName = name;}
    String getName() {return fileName;}
  }

  // checkpoint states
  enum CheckpointStates{START, ROLLED_EDITS, UPLOAD_START, UPLOAD_DONE; }
  /**
   * Implementation of StorageDirType specific to namenode storage
   * A Storage directory could be of type IMAGE which stores only fsimage,
   * or of type EDITS which stores edits or of type IMAGE_AND_EDITS which 
   * stores both fsimage and edits.
   */
  static enum NameNodeDirType implements StorageDirType {
    UNDEFINED,
    IMAGE,
    EDITS,
    IMAGE_AND_EDITS;
    
    public StorageDirType getStorageDirType() {
      return this;
    }
    
    public boolean isOfType(StorageDirType type) {
      if ((this == IMAGE_AND_EDITS) && (type == IMAGE || type == EDITS))
        return true;
      return this == type;
    }
  }

  protected FSNamesystem namesystem = null;
  protected long checkpointTime = -1L;  // The age of the image
  protected FSEditLog editLog = null;
  private boolean isUpgradeFinalized = false;
  protected MD5Hash imageDigest = null;
  protected MD5Hash newImageDigest = null;

  /**
   * flag that controls if we try to restore failed storages
   */
  private boolean restoreFailedStorage = false;

  /**
   * list of failed (and thus removed) storages
   */
  protected List<StorageDirectory> removedStorageDirs = new ArrayList<StorageDirectory>();
    
  /**
   * URIs for importing an image from a checkpoint. In the default case,
   * URIs will represent directories. 
   */
  private Collection<URI> checkpointDirs;
  private Collection<URI> checkpointEditsDirs;

  private Configuration conf;

  /**
   * Can fs-image be rolled?
   */
  volatile protected CheckpointStates ckptState = FSImage.CheckpointStates.START; 

  /**
   */
  FSImage() {
    this((FSNamesystem)null);
  }

  /**
   * Constructor
   * @param conf Configuration
   */
  FSImage(Configuration conf) throws IOException {
    this();
    this.conf = conf; // TODO we have too many constructors, this is a mess

    if(conf.getBoolean(DFSConfigKeys.DFS_NAMENODE_NAME_DIR_RESTORE_KEY, 
        DFSConfigKeys.DFS_NAMENODE_NAME_DIR_RESTORE_DEFAULT)) {
      NameNode.LOG.info("set FSImage.restoreFailedStorage");
      setRestoreFailedStorage(true);
    }
    setCheckpointDirectories(FSImage.getCheckpointDirs(conf, null),
        FSImage.getCheckpointEditsDirs(conf, null));
  }

  private FSImage(FSNamesystem ns) {
    super(NodeType.NAME_NODE);
    this.conf = new Configuration();
    this.editLog = new FSEditLog(this);
    setFSNamesystem(ns);
  }

  /**
   * @throws IOException 
   */
  FSImage(Collection<URI> fsDirs, Collection<URI> fsEditsDirs) 
      throws IOException {
    this();
    setStorageDirectories(fsDirs, fsEditsDirs);
  }

  public FSImage(StorageInfo storageInfo) {
    super(NodeType.NAME_NODE, storageInfo);
  }

  /**
   * Represents an Image (image and edit file).
   * @throws IOException 
   */
  FSImage(URI imageDir) throws IOException {
    this();
    ArrayList<URI> dirs = new ArrayList<URI>(1);
    ArrayList<URI> editsDirs = new ArrayList<URI>(1);
    dirs.add(imageDir);
    editsDirs.add(imageDir);
    setStorageDirectories(dirs, editsDirs);
  }
  
  protected FSNamesystem getFSNamesystem() {
    return namesystem;
  }

  void setFSNamesystem(FSNamesystem ns) {
    namesystem = ns;
  }

  public void setRestoreFailedStorage(boolean val) {
    LOG.info("set restore failed storage to " + val);
    restoreFailedStorage=val;
  }
  
  public boolean getRestoreFailedStorage() {
    return restoreFailedStorage;
  }
  
  void setStorageDirectories(Collection<URI> fsNameDirs,
                             Collection<URI> fsEditsDirs) throws IOException {
    this.storageDirs = new ArrayList<StorageDirectory>();
    this.removedStorageDirs = new ArrayList<StorageDirectory>();
    
   // Add all name dirs with appropriate NameNodeDirType 
    for (URI dirName : fsNameDirs) {
      checkSchemeConsistency(dirName);
      boolean isAlsoEdits = false;
      for (URI editsDirName : fsEditsDirs) {
        if (editsDirName.compareTo(dirName) == 0) {
          isAlsoEdits = true;
          fsEditsDirs.remove(editsDirName);
          break;
        }
      }
      NameNodeDirType dirType = (isAlsoEdits) ?
                          NameNodeDirType.IMAGE_AND_EDITS :
                          NameNodeDirType.IMAGE;
      // Add to the list of storage directories, only if the 
      // URI is of type file://
      if(dirName.getScheme().compareTo(JournalType.FILE.name().toLowerCase()) 
          == 0){
        this.addStorageDir(new StorageDirectory(new File(dirName.getPath()), 
            dirType));
      }
    }
    
    // Add edits dirs if they are different from name dirs
    for (URI dirName : fsEditsDirs) {
      checkSchemeConsistency(dirName);
      // Add to the list of storage directories, only if the 
      // URI is of type file://
      if(dirName.getScheme().compareTo(JournalType.FILE.name().toLowerCase())
          == 0)
        this.addStorageDir(new StorageDirectory(new File(dirName.getPath()), 
                    NameNodeDirType.EDITS));
    }
  }

  /* 
   * Checks the consistency of a URI, in particular if the scheme 
   * is specified and is supported by a concrete implementation 
   */
  static void checkSchemeConsistency(URI u) throws IOException {
    String scheme = u.getScheme();
    // the URI should have a proper scheme
    if(scheme == null)
      throw new IOException("Undefined scheme for " + u);
    else {
      try {
        // the scheme should be enumerated as JournalType
        JournalType.valueOf(scheme.toUpperCase());
      } catch (IllegalArgumentException iae){
        throw new IOException("Unknown scheme " + scheme + 
            ". It should correspond to a JournalType enumeration value");
      }
    }
  };
  
  void setCheckpointDirectories(Collection<URI> dirs,
                                Collection<URI> editsDirs) {
    checkpointDirs = dirs;
    checkpointEditsDirs = editsDirs;
  }
  
  static File getImageFile(StorageDirectory sd, NameNodeFile type) {
    return new File(sd.getCurrentDir(), type.getName());
  }
  
  List<StorageDirectory> getRemovedStorageDirs() {
    return this.removedStorageDirs;
  }
  
  File getEditFile(StorageDirectory sd) {
    return getImageFile(sd, NameNodeFile.EDITS);
  }
  
  File getEditNewFile(StorageDirectory sd) {
    return getImageFile(sd, NameNodeFile.EDITS_NEW);
  }

  Collection<File> getFiles(NameNodeFile type, NameNodeDirType dirType) {
    ArrayList<File> list = new ArrayList<File>();
    Iterator<StorageDirectory> it = (dirType == null) ? dirIterator() :
                                    dirIterator(dirType);
    for ( ;it.hasNext(); ) {
      list.add(getImageFile(it.next(), type));
    }
    return list;
  }

  Collection<URI> getDirectories(NameNodeDirType dirType) 
      throws IOException {
    ArrayList<URI> list = new ArrayList<URI>();
    Iterator<StorageDirectory> it = (dirType == null) ? dirIterator() :
                                    dirIterator(dirType);
    for ( ;it.hasNext(); ) {
      StorageDirectory sd = it.next();
      try {
        list.add(Util.fileAsURI(sd.getRoot()));
      } catch (IOException e) {
        throw new IOException("Exception while processing " +
            "StorageDirectory " + sd.getRoot(), e);
      }
    }
    return list;
  }

  /**
   * Retrieve current directories of type IMAGE
   * @return Collection of URI representing image directories 
   * @throws IOException in case of URI processing error
   */
  Collection<URI> getImageDirectories() throws IOException {
    return getDirectories(NameNodeDirType.IMAGE);
  }

  /**
   * Retrieve current directories of type EDITS
   * @return Collection of URI representing edits directories 
   * @throws IOException in case of URI processing error
   */
  Collection<URI> getEditsDirectories() throws IOException {
    return getDirectories(NameNodeDirType.EDITS);
  }

  /**
   * Return number of storage directories of the given type.
   * @param dirType directory type
   * @return number of storage directories of type dirType
   */
  int getNumStorageDirs(NameNodeDirType dirType) {
    if(dirType == null)
      return getNumStorageDirs();
    Iterator<StorageDirectory> it = dirIterator(dirType);
    int numDirs = 0;
    for(; it.hasNext(); it.next())
      numDirs++;
    return numDirs;
  }

  /**
   * Analyze storage directories.
   * Recover from previous transitions if required. 
   * Perform fs state transition if necessary depending on the namespace info.
   * Read storage info. 
   * 
   * @param dataDirs
   * @param startOpt startup option
   * @throws IOException
   * @return true if the image needs to be saved or false otherwise
   */
  boolean recoverTransitionRead(Collection<URI> dataDirs,
                                Collection<URI> editsDirs,
                                StartupOption startOpt)
      throws IOException {
    assert startOpt != StartupOption.FORMAT : 
      "NameNode formatting should be performed before reading the image";
    
    // none of the data dirs exist
    if((dataDirs.size() == 0 || editsDirs.size() == 0) 
                             && startOpt != StartupOption.IMPORT)  
      throw new IOException(
        "All specified directories are not accessible or do not exist.");
    
    if(startOpt == StartupOption.IMPORT 
        && (checkpointDirs == null || checkpointDirs.isEmpty()))
      throw new IOException("Cannot import image from a checkpoint. "
                          + "\"dfs.namenode.checkpoint.dir\" is not set." );

    if(startOpt == StartupOption.IMPORT 
        && (checkpointEditsDirs == null || checkpointEditsDirs.isEmpty()))
      throw new IOException("Cannot import image from a checkpoint. "
                          + "\"dfs.namenode.checkpoint.dir\" is not set." );
    
    setStorageDirectories(dataDirs, editsDirs);
    // 1. For each data directory calculate its state and 
    // check whether all is consistent before transitioning.
    Map<StorageDirectory, StorageState> dataDirStates = 
             new HashMap<StorageDirectory, StorageState>();
    boolean isFormatted = false;
    for (Iterator<StorageDirectory> it = 
                      dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      StorageState curState;
      try {
        curState = sd.analyzeStorage(startOpt);
        // sd is locked but not opened
        switch(curState) {
        case NON_EXISTENT:
          // name-node fails if any of the configured storage dirs are missing
          throw new InconsistentFSStateException(sd.getRoot(),
                      "storage directory does not exist or is not accessible.");
        case NOT_FORMATTED:
          break;
        case NORMAL:
          break;
        default:  // recovery is possible
          sd.doRecover(curState);      
        }
        if (curState != StorageState.NOT_FORMATTED 
            && startOpt != StartupOption.ROLLBACK) {
          sd.read(); // read and verify consistency with other directories
          isFormatted = true;
        }
        if (startOpt == StartupOption.IMPORT && isFormatted)
          // import of a checkpoint is allowed only into empty image directories
          throw new IOException("Cannot import image from a checkpoint. " 
              + " NameNode already contains an image in " + sd.getRoot());
      } catch (IOException ioe) {
        sd.unlock();
        throw ioe;
      }
      dataDirStates.put(sd,curState);
    }
    
    if (!isFormatted && startOpt != StartupOption.ROLLBACK 
                     && startOpt != StartupOption.IMPORT)
      throw new IOException("NameNode is not formatted.");
    if (layoutVersion < LAST_PRE_UPGRADE_LAYOUT_VERSION) {
      checkVersionUpgradable(layoutVersion);
    }
    if (startOpt != StartupOption.UPGRADE
          && layoutVersion < LAST_PRE_UPGRADE_LAYOUT_VERSION
          && layoutVersion != FSConstants.LAYOUT_VERSION)
        throw new IOException(
           "\nFile system image contains an old layout version " + layoutVersion
         + ".\nAn upgrade to version " + FSConstants.LAYOUT_VERSION
         + " is required.\nPlease restart NameNode with -upgrade option.");
    // check whether distributed upgrade is reguired and/or should be continued
    verifyDistributedUpgradeProgress(startOpt);

    // 2. Format unformatted dirs.
    this.checkpointTime = 0L;
    for (Iterator<StorageDirectory> it = 
                     dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      StorageState curState = dataDirStates.get(sd);
      switch(curState) {
      case NON_EXISTENT:
        throw new IOException(StorageState.NON_EXISTENT + 
                              " state cannot be here");
      case NOT_FORMATTED:
        LOG.info("Storage directory " + sd.getRoot() + " is not formatted.");
        LOG.info("Formatting ...");
        sd.clearDirectory(); // create empty currrent dir
        break;
      default:
        break;
      }
    }

    // 3. Do transitions
    switch(startOpt) {
    case UPGRADE:
      doUpgrade();
      return false; // upgrade saved image already
    case IMPORT:
      doImportCheckpoint();
      return false; // import checkpoint saved image already
    case ROLLBACK:
      doRollback();
      break;
    case REGULAR:
      // just load the image
    }
    
    boolean needToSave = loadFSImage();

    assert editLog != null : "editLog must be initialized";
    if(!editLog.isOpen())
      editLog.open();
    
    return needToSave;
  }

  private void doUpgrade() throws IOException {
    if(getDistributedUpgradeState()) {
      // only distributed upgrade need to continue
      // don't do version upgrade
      this.loadFSImage();
      initializeDistributedUpgrade();
      return;
    }
    // Upgrade is allowed only if there are 
    // no previous fs states in any of the directories
    for (Iterator<StorageDirectory> it = 
                           dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      if (sd.getPreviousDir().exists())
        throw new InconsistentFSStateException(sd.getRoot(),
                                               "previous fs state should not exist during upgrade. "
                                               + "Finalize or rollback first.");
    }

    // load the latest image
    this.loadFSImage();

    // Do upgrade for each directory
    long oldCTime = this.getCTime();
    this.cTime = now();  // generate new cTime for the state
    int oldLV = this.getLayoutVersion();
    this.layoutVersion = FSConstants.LAYOUT_VERSION;
    this.checkpointTime = now();
    for (Iterator<StorageDirectory> it = 
                           dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      LOG.info("Upgrading image directory " + sd.getRoot()
               + ".\n   old LV = " + oldLV
               + "; old CTime = " + oldCTime
               + ".\n   new LV = " + this.getLayoutVersion()
               + "; new CTime = " + this.getCTime());
      File curDir = sd.getCurrentDir();
      File prevDir = sd.getPreviousDir();
      File tmpDir = sd.getPreviousTmp();
      assert curDir.exists() : "Current directory must exist.";
      assert !prevDir.exists() : "prvious directory must not exist.";
      assert !tmpDir.exists() : "prvious.tmp directory must not exist.";
      assert !editLog.isOpen() : "Edits log must not be open.";
      // rename current to tmp
      rename(curDir, tmpDir);
      // save new image
      saveCurrent(sd);
      // rename tmp to previous
      rename(tmpDir, prevDir);
      isUpgradeFinalized = false;
      LOG.info("Upgrade of " + sd.getRoot() + " is complete.");
    }
    initializeDistributedUpgrade();
    editLog.open();
  }

  private void doRollback() throws IOException {
    // Rollback is allowed only if there is 
    // a previous fs states in at least one of the storage directories.
    // Directories that don't have previous state do not rollback
    boolean canRollback = false;
    FSImage prevState = new FSImage(getFSNamesystem());
    prevState.layoutVersion = FSConstants.LAYOUT_VERSION;
    for (Iterator<StorageDirectory> it = 
                       dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      File prevDir = sd.getPreviousDir();
      if (!prevDir.exists()) {  // use current directory then
        LOG.info("Storage directory " + sd.getRoot()
                 + " does not contain previous fs state.");
        sd.read(); // read and verify consistency with other directories
        continue;
      }
      StorageDirectory sdPrev = prevState.new StorageDirectory(sd.getRoot());
      sdPrev.read(sdPrev.getPreviousVersionFile());  // read and verify consistency of the prev dir
      if (prevState.getLayoutVersion() != FSConstants.LAYOUT_VERSION) {
        throw new IOException(
          "Cannot rollback to storage version " +
          prevState.getLayoutVersion() +
          " using this version of the NameNode, which uses storage version " +
          FSConstants.LAYOUT_VERSION + ". " +
          "Please use the previous version of HDFS to perform the rollback.");
      }
      canRollback = true;
    }
    if (!canRollback)
      throw new IOException("Cannot rollback. " 
                            + "None of the storage directories contain previous fs state.");

    // Now that we know all directories are going to be consistent
    // Do rollback for each directory containing previous state
    for (Iterator<StorageDirectory> it = 
                          dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      File prevDir = sd.getPreviousDir();
      if (!prevDir.exists())
        continue;

      LOG.info("Rolling back storage directory " + sd.getRoot()
               + ".\n   new LV = " + prevState.getLayoutVersion()
               + "; new CTime = " + prevState.getCTime());
      File tmpDir = sd.getRemovedTmp();
      assert !tmpDir.exists() : "removed.tmp directory must not exist.";
      // rename current to tmp
      File curDir = sd.getCurrentDir();
      assert curDir.exists() : "Current directory must exist.";
      rename(curDir, tmpDir);
      // rename previous to current
      rename(prevDir, curDir);

      // delete tmp dir
      deleteDir(tmpDir);
      LOG.info("Rollback of " + sd.getRoot()+ " is complete.");
    }
    isUpgradeFinalized = true;
    // check whether name-node can start in regular mode
    verifyDistributedUpgradeProgress(StartupOption.REGULAR);
  }

  private void doFinalize(StorageDirectory sd) throws IOException {
    File prevDir = sd.getPreviousDir();
    if (!prevDir.exists()) { // already discarded
      LOG.info("Directory " + prevDir + " does not exist.");
      LOG.info("Finalize upgrade for " + sd.getRoot()+ " is not required.");
      return;
    }
    LOG.info("Finalizing upgrade for storage directory " 
             + sd.getRoot() + "."
             + (getLayoutVersion()==0 ? "" :
                   "\n   cur LV = " + this.getLayoutVersion()
                   + "; cur CTime = " + this.getCTime()));
    assert sd.getCurrentDir().exists() : "Current directory must exist.";
    final File tmpDir = sd.getFinalizedTmp();
    // rename previous to tmp and remove
    rename(prevDir, tmpDir);
    deleteDir(tmpDir);
    isUpgradeFinalized = true;
    LOG.info("Finalize upgrade for " + sd.getRoot()+ " is complete.");
  }

  /**
   * Load image from a checkpoint directory and save it into the current one.
   * @throws IOException
   */
  void doImportCheckpoint() throws IOException {
    FSNamesystem fsNamesys = getFSNamesystem();
    FSImage ckptImage = new FSImage(fsNamesys);
    // replace real image with the checkpoint image
    FSImage realImage = fsNamesys.getFSImage();
    assert realImage == this;
    fsNamesys.dir.fsImage = ckptImage;
    // load from the checkpoint dirs
    try {
      ckptImage.recoverTransitionRead(checkpointDirs, checkpointEditsDirs,
                                              StartupOption.REGULAR);
    } finally {
      ckptImage.close();
    }
    // return back the real image
    realImage.setStorageInfo(ckptImage);
    checkpointTime = ckptImage.checkpointTime;
    fsNamesys.dir.fsImage = realImage;
    // and save it but keep the same checkpointTime
    saveNamespace(false);
  }

  void finalizeUpgrade() throws IOException {
    for (Iterator<StorageDirectory> it = 
                          dirIterator(); it.hasNext();) {
      doFinalize(it.next());
    }
  }

  boolean isUpgradeFinalized() {
    return isUpgradeFinalized;
  }

  protected void getFields(Properties props, 
                           StorageDirectory sd 
                           ) throws IOException {
    super.getFields(props, sd);
    if (layoutVersion == 0)
      throw new IOException("NameNode directory " 
                            + sd.getRoot() + " is not formatted.");
    String sDUS, sDUV;
    sDUS = props.getProperty("distributedUpgradeState"); 
    sDUV = props.getProperty("distributedUpgradeVersion");
    setDistributedUpgradeState(
        sDUS == null? false : Boolean.parseBoolean(sDUS),
        sDUV == null? getLayoutVersion() : Integer.parseInt(sDUV));
    
    String sMd5 = props.getProperty(MESSAGE_DIGEST_PROPERTY);
    if (LayoutVersion.supports(Feature.FSIMAGE_CHECKSUM, layoutVersion)) {
    
      if (sMd5 == null) {
        throw new InconsistentFSStateException(sd.getRoot(),
            "file " + STORAGE_FILE_VERSION + " does not have MD5 image digest.");
      }
      this.imageDigest = new MD5Hash(sMd5);
    } else if (sMd5 != null) {
      throw new InconsistentFSStateException(sd.getRoot(),
          "file " + STORAGE_FILE_VERSION +
          " has image MD5 digest when version is " + layoutVersion);
    }

    this.checkpointTime = readCheckpointTime(sd);
  }

  /**
   * Determine the checkpoint time of the specified StorageDirectory
   * 
   * @param sd StorageDirectory to check
   * @return If file exists and can be read, last checkpoint time. If not, 0L.
   * @throws IOException On errors processing file pointed to by sd
   */
  long readCheckpointTime(StorageDirectory sd) throws IOException {
    File timeFile = getImageFile(sd, NameNodeFile.TIME);
    long timeStamp = 0L;
    if (timeFile.exists() && timeFile.canRead()) {
      DataInputStream in = new DataInputStream(new FileInputStream(timeFile));
      try {
        timeStamp = in.readLong();
      } finally {
        in.close();
      }
    }
    return timeStamp;
  }

  /**
   * Write last checkpoint time and version file into the storage directory.
   * 
   * The version file should always be written last.
   * Missing or corrupted version file indicates that 
   * the checkpoint is not valid.
   * 
   * @param sd storage directory
   * @throws IOException
   */
  protected void setFields(Properties props, 
                           StorageDirectory sd 
                           ) throws IOException {
    super.setFields(props, sd);
    boolean uState = getDistributedUpgradeState();
    int uVersion = getDistributedUpgradeVersion();
    if(uState && uVersion != getLayoutVersion()) {
      props.setProperty("distributedUpgradeState", Boolean.toString(uState));
      props.setProperty("distributedUpgradeVersion", Integer.toString(uVersion)); 
    }
    if (LayoutVersion.supports(Feature.FSIMAGE_CHECKSUM, layoutVersion)) {
      // Though the current NN supports this feature, this function
      // is called with old layoutVersions from the upgrade tests.
      if (imageDigest == null) {
        imageDigest = MD5Hash.digest(
            new FileInputStream(getImageFile(sd, NameNodeFile.IMAGE)));
      }
      props.setProperty(MESSAGE_DIGEST_PROPERTY, imageDigest.toString());
    }

    writeCheckpointTime(sd);
  }

  /**
   * Write last checkpoint time into a separate file.
   * 
   * @param sd
   * @throws IOException
   */
  void writeCheckpointTime(StorageDirectory sd) throws IOException {
    if (checkpointTime < 0L)
      return; // do not write negative time
    File timeFile = getImageFile(sd, NameNodeFile.TIME);
    if (timeFile.exists() && ! timeFile.delete()) {
        LOG.error("Cannot delete chekpoint time file: "
                  + timeFile.getCanonicalPath());
    }
    FileOutputStream fos = new FileOutputStream(timeFile);
    DataOutputStream out = new DataOutputStream(fos);
    try {
      out.writeLong(checkpointTime);
      out.flush();
      fos.getChannel().force(true);
    } finally {
      out.close();
    }
  }

  /**
   * Record new checkpoint time in order to
   * distinguish healthy directories from the removed ones.
   * If there is an error writing new checkpoint time, the corresponding
   * storage directory is removed from the list.
   */
  void incrementCheckpointTime() {
    setCheckpointTime(checkpointTime + 1);
  }

  /**
   * The age of the namespace state.<p>
   * Reflects the latest time the image was saved.
   * Modified with every save or a checkpoint.
   * Persisted in VERSION file.
   */
  long getCheckpointTime() {
    return checkpointTime;
  }

  void setCheckpointTime(long newCpT) {
    checkpointTime = newCpT;
    ArrayList<StorageDirectory> failingStorageDirs = new ArrayList<StorageDirectory>(1);
    // Write new checkpoint time in all storage directories
    for(Iterator<StorageDirectory> it =
                          dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      try {
        writeCheckpointTime(sd);
      } catch(IOException e) {
        // Close any edits stream associated with this dir and remove directory
        LOG.warn("incrementCheckpointTime failed on " + sd.getRoot().getPath() + ";type="+sd.getStorageDirType());
        // Since writeCheckpointTime may also encounter an IOException in case
        // underlying storage fails
        failingStorageDirs.add(sd);
      }
    }
    if (failingStorageDirs.size() > 0)
      processIOError(failingStorageDirs, true);
  }

  /**
   * @param sds - array of SDs to process
   * @param propagate - flag, if set - then call corresponding EditLog stream's 
   * processIOError function.
   */
  void processIOError(List<StorageDirectory> sds, boolean propagate) {
    ArrayList<EditLogOutputStream> al = null;
    synchronized (sds) {
      for (StorageDirectory sd : sds) {
        // if has a stream assosiated with it - remove it too..
        if (propagate && sd.getStorageDirType().isOfType(NameNodeDirType.EDITS)) {
          EditLogOutputStream eStream = editLog.getEditsStream(sd);
          if (al == null)
            al = new ArrayList<EditLogOutputStream>(1);
          al.add(eStream);
        }

        for (Iterator<StorageDirectory> it = dirIterator(); it.hasNext();) {
          StorageDirectory sd1 = it.next();
          if (sd.equals(sd1)) {
            // add storage to the removed list
            LOG.warn("FSImage:processIOError: removing storage: "
                + sd.getRoot().getPath());
            try {
              sd1.unlock(); // unlock before removing (in case it will be
                            // restored)
            } catch (Exception e) {
              // nothing
            }
            removedStorageDirs.add(sd1);
            it.remove();
            break;
          }
        }
      }
    }
    // if there are some edit log streams to remove    
    if(propagate && al != null) 
      editLog.processIOError(al, false);
    
    //if called from edits log, the it will call increment from there
    if(propagate) incrementCheckpointTime(); 
  }

  public FSEditLog getEditLog() {
    return editLog;
  }

  public boolean isPreUpgradableLayout(StorageDirectory sd) throws IOException {
    File oldImageDir = new File(sd.getRoot(), "image");
    if (!oldImageDir.exists()) {
      return false;
    }
    // check the layout version inside the image file
    File oldF = new File(oldImageDir, "fsimage");
    RandomAccessFile oldFile = new RandomAccessFile(oldF, "rws");
    try {
      oldFile.seek(0);
      int oldVersion = oldFile.readInt();
      if (oldVersion < LAST_PRE_UPGRADE_LAYOUT_VERSION)
        return false;
    } finally {
      oldFile.close();
    }
    return true;
  }

  //
  // Atomic move sequence, to recover from interrupted checkpoint
  //
  boolean recoverInterruptedCheckpoint(StorageDirectory nameSD,
                                       StorageDirectory editsSD) 
                                       throws IOException {
    boolean needToSave = false;
    File curFile = getImageFile(nameSD, NameNodeFile.IMAGE);
    File ckptFile = getImageFile(nameSD, NameNodeFile.IMAGE_NEW);

    //
    // If we were in the midst of a checkpoint
    //
    if (ckptFile.exists()) {
      needToSave = true;
      if (getImageFile(editsSD, NameNodeFile.EDITS_NEW).exists()) {
        //
        // checkpointing migth have uploaded a new
        // merged image, but we discard it here because we are
        // not sure whether the entire merged image was uploaded
        // before the namenode crashed.
        //
        if (!ckptFile.delete()) {
          throw new IOException("Unable to delete " + ckptFile);
        }
      } else {
        //
        // checkpointing was in progress when the namenode
        // shutdown. The fsimage.ckpt was created and the edits.new
        // file was moved to edits. We complete that checkpoint by
        // moving fsimage.new to fsimage. There is no need to 
        // update the fstime file here. renameTo fails on Windows
        // if the destination file already exists.
        //
        if (!ckptFile.renameTo(curFile)) {
          if (!curFile.delete())
            LOG.warn("Unable to delete dir " + curFile + " before rename");
          if (!ckptFile.renameTo(curFile)) {
            throw new IOException("Unable to rename " + ckptFile +
                                  " to " + curFile);
          }
        }
      }
    }
    return needToSave;
  }

  /**
   * Choose latest image from one of the directories,
   * load it and merge with the edits from that directory.
   * 
   * Saving and loading fsimage should never trigger symlink resolution. 
   * The paths that are persisted do not have *intermediate* symlinks 
   * because intermediate symlinks are resolved at the time files, 
   * directories, and symlinks are created. All paths accessed while 
   * loading or saving fsimage should therefore only see symlinks as 
   * the final path component, and the functions called below do not
   * resolve symlinks that are the final path component.
   *
   * @return whether the image should be saved
   * @throws IOException
   */
  boolean loadFSImage() throws IOException {
    long latestNameCheckpointTime = Long.MIN_VALUE;
    long latestEditsCheckpointTime = Long.MIN_VALUE;
    boolean needToSave = false;
    isUpgradeFinalized = true;
    
    StorageDirectory latestNameSD = null;
    StorageDirectory latestEditsSD = null;
    
    Collection<String> imageDirs = new ArrayList<String>();
    Collection<String> editsDirs = new ArrayList<String>();
    
    // Set to determine if all of storageDirectories share the same checkpoint
    Set<Long> checkpointTimes = new HashSet<Long>();

    // Process each of the storage directories to find the pair of
    // newest image file and edit file
    for (Iterator<StorageDirectory> it = dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();

      // Was the file just formatted?
      if (!sd.getVersionFile().exists()) {
        needToSave |= true;
        continue;
      }
      
      boolean imageExists = false;
      boolean editsExists = false;
      
      // Determine if sd is image, edits or both
      if (sd.getStorageDirType().isOfType(NameNodeDirType.IMAGE)) {
        imageExists = getImageFile(sd, NameNodeFile.IMAGE).exists();
        imageDirs.add(sd.getRoot().getCanonicalPath());
      }
      
      if (sd.getStorageDirType().isOfType(NameNodeDirType.EDITS)) {
        editsExists = getImageFile(sd, NameNodeFile.EDITS).exists();
        editsDirs.add(sd.getRoot().getCanonicalPath());
      }
      
      checkpointTime = readCheckpointTime(sd);

      checkpointTimes.add(checkpointTime);
      
      if (sd.getStorageDirType().isOfType(NameNodeDirType.IMAGE) && 
         (latestNameCheckpointTime < checkpointTime) && imageExists) {
        latestNameCheckpointTime = checkpointTime;
        latestNameSD = sd;
      }
      
      if (sd.getStorageDirType().isOfType(NameNodeDirType.EDITS) && 
           (latestEditsCheckpointTime < checkpointTime) && editsExists) {
        latestEditsCheckpointTime = checkpointTime;
        latestEditsSD = sd;
      }
      
      // check that we have a valid, non-default checkpointTime
      if (checkpointTime <= 0L)
        needToSave |= true;
      
      // set finalized flag
      isUpgradeFinalized = isUpgradeFinalized && !sd.getPreviousDir().exists();
    }

    // We should have at least one image and one edits dirs
    if (latestNameSD == null)
      throw new IOException("Image file is not found in " + imageDirs);
    if (latestEditsSD == null)
      throw new IOException("Edits file is not found in " + editsDirs);

    // Make sure we are loading image and edits from same checkpoint
    if (latestNameCheckpointTime > latestEditsCheckpointTime
        && latestNameSD != latestEditsSD
        && latestNameSD.getStorageDirType() == NameNodeDirType.IMAGE
        && latestEditsSD.getStorageDirType() == NameNodeDirType.EDITS) {
      // This is a rare failure when NN has image-only and edits-only
      // storage directories, and fails right after saving images,
      // in some of the storage directories, but before purging edits.
      // See -NOTE- in saveNamespace().
      LOG.error("This is a rare failure scenario!!!");
      LOG.error("Image checkpoint time " + latestNameCheckpointTime +
                " > edits checkpoint time " + latestEditsCheckpointTime);
      LOG.error("Name-node will treat the image as the latest state of " +
                "the namespace. Old edits will be discarded.");
    } else if (latestNameCheckpointTime != latestEditsCheckpointTime)
      throw new IOException("Inconsistent storage detected, " +
                      "image and edits checkpoint times do not match. " +
                      "image checkpoint time = " + latestNameCheckpointTime +
                      "edits checkpoint time = " + latestEditsCheckpointTime);
    
    // If there was more than one checkpointTime recorded we should save
    needToSave |= checkpointTimes.size() != 1;
    
    // Recover from previous interrupted checkpoint, if any
    needToSave |= recoverInterruptedCheckpoint(latestNameSD, latestEditsSD);

    //
    // Load in bits
    //
    latestNameSD.read();
    needToSave |= loadFSImage(getImageFile(latestNameSD, NameNodeFile.IMAGE));
    
    // Load latest edits
    if (latestNameCheckpointTime > latestEditsCheckpointTime)
      // the image is already current, discard edits
      needToSave |= true;
    else // latestNameCheckpointTime == latestEditsCheckpointTime
      needToSave |= (loadFSEdits(latestEditsSD) > 0);
    
    return needToSave;
  }

  /**
   * Load in the filesystem image from file. It's a big list of
   * filenames and blocks.  Return whether we should
   * "re-save" and consolidate the edit-logs
   */
  boolean loadFSImage(File curFile) throws IOException {
    FSImageFormat.Loader loader = new FSImageFormat.Loader(conf);
    loader.load(curFile, getFSNamesystem());


    // Check that the image digest we loaded matches up with what
    // we expected
    MD5Hash readImageMd5 = loader.getLoadedImageMd5();
    if (imageDigest == null) {
      imageDigest = readImageMd5; // set this fsimage's checksum
    } else if (!imageDigest.equals(readImageMd5)) {
      throw new IOException("Image file " + curFile +
          " is corrupt with MD5 checksum of " + readImageMd5 +
          " but expecting " + imageDigest);
    }

    this.namespaceID = loader.getLoadedNamespaceID();
    this.layoutVersion = loader.getLoadedImageVersion();

    boolean needToSave =
      loader.getLoadedImageVersion() != FSConstants.LAYOUT_VERSION;
    return needToSave;
  }

  /**
   * Load and merge edits from two edits files
   * 
   * @param sd storage directory
   * @return number of edits loaded
   * @throws IOException
   */
  int loadFSEdits(StorageDirectory sd) throws IOException {
    FSEditLogLoader loader = new FSEditLogLoader(namesystem);
    
    int numEdits = 0;
    EditLogFileInputStream edits = 
      new EditLogFileInputStream(getImageFile(sd, NameNodeFile.EDITS));
    
    numEdits = loader.loadFSEdits(edits);
    edits.close();
    File editsNew = getImageFile(sd, NameNodeFile.EDITS_NEW);
    
    if (editsNew.exists() && editsNew.length() > 0) {
      edits = new EditLogFileInputStream(editsNew);
      numEdits += loader.loadFSEdits(edits);
      edits.close();
    }
    if (numEdits == 0 && editsNew.exists()) {
      numEdits++;
    }
    // update the counts.
    getFSNamesystem().dir.updateCountForINodeWithQuota();    
    
    return numEdits;
  }

  /**
   * Save the contents of the FS image to the file.
   */
  void saveFSImage(File newFile) throws IOException {
    FSImageFormat.Saver saver = new FSImageFormat.Saver();
    FSImageCompression compression = FSImageCompression.createCompression(conf);
    saver.save(newFile, getFSNamesystem(), compression);
    setImageDigest(saver.getSavedDigest());
  }

  public void setImageDigest(MD5Hash digest) {
    this.imageDigest = digest;
  }
  /**
   * FSImageSaver is being run in a separate thread when saving
   * FSImage. There is one thread per each copy of the image.
   *
   * FSImageSaver assumes that it was launched from a thread that holds
   * FSNamesystem lock and waits for the execution of FSImageSaver thread
   * to finish.
   * This way we are guraranteed that the namespace is not being updated
   * while multiple instances of FSImageSaver are traversing it
   * and writing it out.
   */
  private class FSImageSaver implements Runnable {
    private StorageDirectory sd;
    private List<StorageDirectory> errorSDs;
    
    FSImageSaver(StorageDirectory sd, List<StorageDirectory> errorSDs) {
      this.sd = sd;
      this.errorSDs = errorSDs;
    }
    
    public void run() {
      try {
        saveCurrent(sd);
      } catch (Throwable t) {
        LOG.error("Unable to save image for " + sd.getRoot(), t);
        errorSDs.add(sd);
      }
    }
    
    public String toString() {
      return "FSImageSaver for " + sd.getRoot() +
             " of type " + sd.getStorageDirType();
    }
  }
  
  private void waitForThreads(List<Thread> threads) {
    for (Thread thread : threads) {
      while (thread.isAlive()) {
        try {
          thread.join();
        } catch (InterruptedException iex) {
          LOG.error("Caught exception while waiting for thread " +
                    thread.getName() + " to finish. Retrying join");
        }        
      }
    }
  }
  /**
   * Save the contents of the FS image and create empty edits.
   * 
   * In order to minimize the recovery effort in case of failure during
   * saveNamespace the algorithm reduces discrepancy between directory states
   * by performing updates in the following order:
   * <ol>
   * <li> rename current to lastcheckpoint.tmp for all of them,</li>
   * <li> save image and recreate edits for all of them,</li>
   * <li> rename lastcheckpoint.tmp to previous.checkpoint.</li>
   * </ol>
   * On stage (2) we first save all images, then recreate edits.
   * Otherwise the name-node may purge all edits and fail,
   * in which case the journal will be lost.
   */
  void saveNamespace(boolean renewCheckpointTime) throws IOException {
    assert editLog != null : "editLog must be initialized";
    editLog.close();
    if(renewCheckpointTime)
      this.checkpointTime = now();
    List<StorageDirectory> errorSDs =
      Collections.synchronizedList(new ArrayList<StorageDirectory>());

    // mv current -> lastcheckpoint.tmp
    for (Iterator<StorageDirectory> it = dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      try {
        moveCurrent(sd);
      } catch(IOException ie) {
        LOG.error("Unable to move current for " + sd.getRoot(), ie);
        errorSDs.add(sd);
      }
    }

    List<Thread> saveThreads = new ArrayList<Thread>();
    // save images into current
    for (Iterator<StorageDirectory> it = dirIterator(NameNodeDirType.IMAGE);
                                                              it.hasNext();) {
      StorageDirectory sd = it.next();
      if (errorSDs.contains(sd)) {
        continue;
      }
      try {
        FSImageSaver saver = new FSImageSaver(sd, errorSDs);
        Thread saveThread = new Thread(saver, saver.toString());
        saveThreads.add(saveThread);
        saveThread.start();
      } catch (Exception e) {
        LOG.error("Failed save to image directory " + sd.getRoot(), e);
        errorSDs.add(sd);
        continue;
      }
    }
    waitForThreads(saveThreads);
    saveThreads.clear();

    // -NOTE-
    // If NN has image-only and edits-only storage directories and fails here 
    // the image will have the latest namespace state.
    // During startup the image-only directories will recover by discarding
    // lastcheckpoint.tmp, while
    // the edits-only directories will recover by falling back
    // to the old state contained in their lastcheckpoint.tmp.
    // The edits directories should be discarded during startup because their
    // checkpointTime is older than that of image directories.
    // recreate edits in current
    for (Iterator<StorageDirectory> it = dirIterator(NameNodeDirType.EDITS);
                                                              it.hasNext();) {
      final StorageDirectory sd = it.next();
      if (errorSDs.contains(sd)) {
        continue;
      }
      try {
        FSImageSaver saver = new FSImageSaver(sd, errorSDs);
        Thread saveThread = new Thread(saver, saver.toString());
        saveThreads.add(saveThread);
        saveThread.start();
      } catch (Exception e) {
        LOG.error("Failed save to edits directory " + sd.getRoot(), e);
        errorSDs.add(sd);
        continue;
      }
    }
    waitForThreads(saveThreads);

    // mv lastcheckpoint.tmp -> previous.checkpoint
    for (Iterator<StorageDirectory> it = dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      if (errorSDs.contains(sd)) {
        continue;
      }
      try {
        moveLastCheckpoint(sd);
      } catch(IOException ie) {
        LOG.error("Unable to move last checkpoint for " + sd.getRoot(), ie);
        errorSDs.add(sd);
        continue;
      }
    }
    
    try {
      processIOError(errorSDs, false);

      // If there was an error in every storage dir, each one will have been
      // removed from the list of storage directories.
      if (getNumStorageDirs(NameNodeDirType.IMAGE) == 0 ||
          getNumStorageDirs(NameNodeDirType.EDITS) == 0) {
        throw new IOException("Failed to save any storage directories while saving namespace");
      }

      if(!editLog.isOpen()) editLog.open();
    } finally {
      ckptState = CheckpointStates.UPLOAD_DONE;
    }
  }

  /**
   * Save current image and empty journal into {@code current} directory.
   */
  protected void saveCurrent(StorageDirectory sd) throws IOException {
    if (getLayoutVersion() != FSConstants.LAYOUT_VERSION) {
      throw new IllegalStateException(
        "NN with storage version " + FSConstants.LAYOUT_VERSION  +
        "cannot save an image with version " + getLayoutVersion());
    }
    File curDir = sd.getCurrentDir();
    NameNodeDirType dirType = (NameNodeDirType)sd.getStorageDirType();
    // save new image or new edits
    if (!curDir.exists() && !curDir.mkdir())
      throw new IOException("Cannot create directory " + curDir);
    if (dirType.isOfType(NameNodeDirType.IMAGE))
      saveFSImage(getImageFile(sd, NameNodeFile.IMAGE));
    if (dirType.isOfType(NameNodeDirType.EDITS))
      editLog.createEditLogFile(getImageFile(sd, NameNodeFile.EDITS));
    // write version and time files
    sd.write();
  }

  /**
   * Move {@code current} to {@code lastcheckpoint.tmp} and
   * recreate empty {@code current}.
   * {@code current} is moved only if it is well formatted,
   * that is contains VERSION file.
   * 
   * @see org.apache.hadoop.hdfs.server.common.Storage.StorageDirectory#getLastCheckpointTmp()
   * @see org.apache.hadoop.hdfs.server.common.Storage.StorageDirectory#getPreviousCheckpoint()
   */
  protected void moveCurrent(StorageDirectory sd)
  throws IOException {
    File curDir = sd.getCurrentDir();
    File tmpCkptDir = sd.getLastCheckpointTmp();
    // mv current -> lastcheckpoint.tmp
    // only if current is formatted - has VERSION file
    if(sd.getVersionFile().exists()) {
      assert curDir.exists() : curDir + " directory must exist.";
      assert !tmpCkptDir.exists() : tmpCkptDir + " directory must not exist.";
      rename(curDir, tmpCkptDir);
    }
    // recreate current
    if(!curDir.exists() && !curDir.mkdir())
      throw new IOException("Cannot create directory " + curDir);
  }

  /**
   * Move {@code lastcheckpoint.tmp} to {@code previous.checkpoint}
   * 
   * @see org.apache.hadoop.hdfs.server.common.Storage.StorageDirectory#getPreviousCheckpoint()
   * @see org.apache.hadoop.hdfs.server.common.Storage.StorageDirectory#getLastCheckpointTmp()
   */
  protected void moveLastCheckpoint(StorageDirectory sd)
  throws IOException {
    File tmpCkptDir = sd.getLastCheckpointTmp();
    File prevCkptDir = sd.getPreviousCheckpoint();
    // remove previous.checkpoint
    if (prevCkptDir.exists())
      deleteDir(prevCkptDir);
    // mv lastcheckpoint.tmp -> previous.checkpoint
    if(tmpCkptDir.exists())
      rename(tmpCkptDir, prevCkptDir);
  }

  /**
   * Generate new namespaceID.
   * 
   * namespaceID is a persistent attribute of the namespace.
   * It is generated when the namenode is formatted and remains the same
   * during the life cycle of the namenode.
   * When a datanodes register they receive it as the registrationID,
   * which is checked every time the datanode is communicating with the 
   * namenode. Datanodes that do not 'know' the namespaceID are rejected.
   * 
   * @return new namespaceID
   */
  private int newNamespaceID() {
    Random r = new Random();
    r.setSeed(now());
    int newID = 0;
    while(newID == 0)
      newID = r.nextInt(0x7FFFFFFF);  // use 31 bits only
    return newID;
  }

  /** Create new dfs name directory.  Caution: this destroys all files
   * in this filesystem. */
  void format(StorageDirectory sd) throws IOException {
    sd.clearDirectory(); // create currrent dir
    sd.lock();
    try {
      saveCurrent(sd);
    } finally {
      sd.unlock();
    }
    LOG.info("Storage directory " + sd.getRoot()
             + " has been successfully formatted.");
  }

  public void format() throws IOException {
    this.layoutVersion = FSConstants.LAYOUT_VERSION;
    this.namespaceID = newNamespaceID();
    this.cTime = 0L;
    this.checkpointTime = now();
    for (Iterator<StorageDirectory> it = 
                           dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      format(sd);
    }
  }


  /**
   * Moves fsimage.ckpt to fsImage and edits.new to edits
   * Reopens the new edits file.
   */
  void rollFSImage(CheckpointSignature sig, 
      boolean renewCheckpointTime) throws IOException {
    sig.validateStorageInfo(this);
    rollFSImage(true);
  }

  private void rollFSImage(boolean renewCheckpointTime)
  throws IOException {
    if (ckptState != CheckpointStates.UPLOAD_DONE
      && !(ckptState == CheckpointStates.ROLLED_EDITS
      && getNumStorageDirs(NameNodeDirType.IMAGE) == 0)) {
      throw new IOException("Cannot roll fsImage before rolling edits log.");
    }

    for (Iterator<StorageDirectory> it = 
                       dirIterator(NameNodeDirType.IMAGE); it.hasNext();) {
      StorageDirectory sd = it.next();
      File ckpt = getImageFile(sd, NameNodeFile.IMAGE_NEW);
      if (!ckpt.exists()) {
        throw new IOException("Checkpoint file " + ckpt +
                              " does not exist");
      }
    }
    editLog.purgeEditLog(); // renamed edits.new to edits
    if(LOG.isDebugEnabled()) {
      LOG.debug("rollFSImage after purgeEditLog: storageList=" + listStorageDirectories());
    }
    //
    // Renames new image
    //
    renameCheckpoint();
    resetVersion(renewCheckpointTime, newImageDigest);
  }

  /**
   * Renames new image
   */
  void renameCheckpoint() {
    ArrayList<StorageDirectory> al = null;
    for (Iterator<StorageDirectory> it = 
      dirIterator(NameNodeDirType.IMAGE); it.hasNext();) {
      StorageDirectory sd = it.next();
      File ckpt = getImageFile(sd, NameNodeFile.IMAGE_NEW);
      File curFile = getImageFile(sd, NameNodeFile.IMAGE);
      // renameTo fails on Windows if the destination file 
      // already exists.
      if(LOG.isDebugEnabled()) {
        LOG.debug("renaming  " + ckpt.getAbsolutePath() + " to "  + curFile.getAbsolutePath());
      }
      if (!ckpt.renameTo(curFile)) {
        if (!curFile.delete() || !ckpt.renameTo(curFile)) {
          LOG.warn("renaming  " + ckpt.getAbsolutePath() + " to "  + 
              curFile.getAbsolutePath() + " FAILED");

          if(al == null) al = new ArrayList<StorageDirectory> (1);
          al.add(sd);
        }
      }
    }
    if(al != null) processIOError(al, true);
  }

  /**
   * Updates version and fstime files in all directories (fsimage and edits).
   */
  void resetVersion(boolean renewCheckpointTime, MD5Hash newImageDigest) throws IOException {
    this.layoutVersion = FSConstants.LAYOUT_VERSION;
    if(renewCheckpointTime)
      this.checkpointTime = now();
    this.imageDigest = newImageDigest;
    
    ArrayList<StorageDirectory> al = null;
    for (Iterator<StorageDirectory> it = 
                           dirIterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      // delete old edits if sd is the image only the directory
      if (!sd.getStorageDirType().isOfType(NameNodeDirType.EDITS)) {
        File editsFile = getImageFile(sd, NameNodeFile.EDITS);
        if(editsFile.exists() && !editsFile.delete())
          throw new IOException("Cannot delete edits file " 
                                + editsFile.getCanonicalPath());
      }
      // delete old fsimage if sd is the edits only the directory
      if (!sd.getStorageDirType().isOfType(NameNodeDirType.IMAGE)) {
        File imageFile = getImageFile(sd, NameNodeFile.IMAGE);
        if(imageFile.exists() && !imageFile.delete())
          throw new IOException("Cannot delete image file " 
                                + imageFile.getCanonicalPath());
      }
      try {
        sd.write();
      } catch (IOException e) {
        LOG.error("Cannot write file " + sd.getRoot(), e);
        
        if(al == null) al = new ArrayList<StorageDirectory> (1);
        al.add(sd);       
      }
    }
    if(al != null) processIOError(al, true);
    ckptState = FSImage.CheckpointStates.START;
  }

  CheckpointSignature rollEditLog() throws IOException {
    getEditLog().rollEditLog();
    ckptState = CheckpointStates.ROLLED_EDITS;
    // If checkpoint fails this should be the most recent image, therefore
    incrementCheckpointTime();
    return new CheckpointSignature(this);
  }

  /**
   * This is called just before a new checkpoint is uploaded to the
   * namenode.
   */
  void validateCheckpointUpload(CheckpointSignature sig) throws IOException {
    if (ckptState != CheckpointStates.ROLLED_EDITS) {
      throw new IOException("Namenode is not expecting an new image " +
                             ckptState);
    } 
    // verify token
    long modtime = getEditLog().getFsEditTime();
    if (sig.editsTime != modtime) {
      throw new IOException("Namenode has an edit log with timestamp of " +
                            DATE_FORM.format(new Date(modtime)) +
                            " but new checkpoint was created using editlog " +
                            " with timestamp " + 
                            DATE_FORM.format(new Date(sig.editsTime)) + 
                            ". Checkpoint Aborted.");
    }
    sig.validateStorageInfo(this);
    ckptState = FSImage.CheckpointStates.UPLOAD_START;
  }

  /**
   * Start checkpoint.
   * <p>
   * If backup storage contains image that is newer than or incompatible with 
   * what the active name-node has, then the backup node should shutdown.<br>
   * If the backup image is older than the active one then it should 
   * be discarded and downloaded from the active node.<br>
   * If the images are the same then the backup image will be used as current.
   * 
   * @param bnReg the backup node registration.
   * @param nnReg this (active) name-node registration.
   * @return {@link NamenodeCommand} if backup node should shutdown or
   * {@link CheckpointCommand} prescribing what backup node should 
   *         do with its image.
   * @throws IOException
   */
  NamenodeCommand startCheckpoint(NamenodeRegistration bnReg, // backup node
                                  NamenodeRegistration nnReg) // active name-node
  throws IOException {
    String msg = null;
    // Verify that checkpoint is allowed
    if(bnReg.getNamespaceID() != this.getNamespaceID())
      msg = "Name node " + bnReg.getAddress()
            + " has incompatible namespace id: " + bnReg.getNamespaceID()
            + " expected: " + getNamespaceID();
    else if(bnReg.isRole(NamenodeRole.ACTIVE))
      msg = "Name node " + bnReg.getAddress()
            + " role " + bnReg.getRole() + ": checkpoint is not allowed.";
    else if(bnReg.getLayoutVersion() < this.getLayoutVersion()
        || (bnReg.getLayoutVersion() == this.getLayoutVersion()
            && bnReg.getCTime() > this.getCTime())
        || (bnReg.getLayoutVersion() == this.getLayoutVersion()
            && bnReg.getCTime() == this.getCTime()
            && bnReg.getCheckpointTime() > this.checkpointTime))
      // remote node has newer image age
      msg = "Name node " + bnReg.getAddress()
            + " has newer image layout version: LV = " +bnReg.getLayoutVersion()
            + " cTime = " + bnReg.getCTime()
            + " checkpointTime = " + bnReg.getCheckpointTime()
            + ". Current version: LV = " + getLayoutVersion()
            + " cTime = " + getCTime()
            + " checkpointTime = " + checkpointTime;
    if(msg != null) {
      LOG.error(msg);
      return new NamenodeCommand(NamenodeProtocol.ACT_SHUTDOWN);
    }
    boolean isImgObsolete = true;
    if(bnReg.getLayoutVersion() == this.getLayoutVersion()
        && bnReg.getCTime() == this.getCTime()
        && bnReg.getCheckpointTime() == this.checkpointTime)
      isImgObsolete = false;
    boolean needToReturnImg = true;
    if(getNumStorageDirs(NameNodeDirType.IMAGE) == 0)
      // do not return image if there are no image directories
      needToReturnImg = false;
    CheckpointSignature sig = rollEditLog();
    getEditLog().logJSpoolStart(bnReg, nnReg);
    return new CheckpointCommand(sig, isImgObsolete, needToReturnImg);
  }

  /**
   * End checkpoint.
   * <p>
   * Rename uploaded checkpoint to the new image;
   * purge old edits file;
   * rename edits.new to edits;
   * redirect edit log streams to the new edits;
   * update checkpoint time if the remote node is a checkpoint only node.
   * 
   * @param sig
   * @param remoteNNRole
   * @throws IOException
   */
  void endCheckpoint(CheckpointSignature sig,
                     NamenodeRole remoteNNRole) throws IOException {
    sig.validateStorageInfo(this);
    // Renew checkpoint time for the active if the other is a checkpoint-node.
    // The checkpoint-node should have older image for the next checkpoint 
    // to take effect.
    // The backup-node always has up-to-date image and will have the same
    // checkpoint time as the active node.
    boolean renewCheckpointTime = remoteNNRole.equals(NamenodeRole.CHECKPOINT);
    rollFSImage(sig, renewCheckpointTime);
  }

  CheckpointStates getCheckpointState() {
    return ckptState;
  }

  void setCheckpointState(CheckpointStates cs) {
    ckptState = cs;
  }

  /**
   * This is called when a checkpoint upload finishes successfully.
   */
  synchronized void checkpointUploadDone() {
    ckptState = CheckpointStates.UPLOAD_DONE;
  }

  synchronized void close() throws IOException {
    getEditLog().close();
    unlockAll();
  }

  /**
   * Return the name of the image file.
   */
  File getFsImageName() {
    StorageDirectory sd = null;
    for (Iterator<StorageDirectory> it = 
      dirIterator(NameNodeDirType.IMAGE); it.hasNext();) {
      sd = it.next();
      File fsImage = getImageFile(sd, NameNodeFile.IMAGE);
      if(sd.getRoot().canRead() && fsImage.exists())
        return fsImage;
    }
    return null;
  }

  /**
   * See if any of removed storages iw "writable" again, and can be returned 
   * into service
   */
  synchronized void attemptRestoreRemovedStorage() {   
    // if directory is "alive" - copy the images there...
    if(!restoreFailedStorage || removedStorageDirs.size() == 0) 
      return; //nothing to restore
    
    LOG.info("FSImage.attemptRestoreRemovedStorage: check removed(failed) " +
        "storarge. removedStorages size = " + removedStorageDirs.size());
    for(Iterator<StorageDirectory> it = this.removedStorageDirs.iterator(); it.hasNext();) {
      StorageDirectory sd = it.next();
      File root = sd.getRoot();
      LOG.info("currently disabled dir " + root.getAbsolutePath() + 
          "; type="+sd.getStorageDirType() + ";canwrite="+root.canWrite());
      try {
        
        if(root.exists() && root.canWrite()) { 
          // when we try to restore we just need to remove all the data
          // without saving current in-memory state (which could've changed).
          sd.clearDirectory();
          
          LOG.info("restoring dir " + sd.getRoot().getAbsolutePath());
          if(sd.getStorageDirType().isOfType(NameNodeDirType.EDITS)) {
            File eFile = getEditFile(sd);
            editLog.addNewEditLogStream(eFile);
          }
          this.addStorageDir(sd); // restore
          it.remove();
        }
      } catch(IOException e) {
        LOG.warn("failed to restore " + sd.getRoot().getAbsolutePath(), e);
      }
    }    
  }
  
  public File getFsEditName() throws IOException {
    return getEditLog().getFsEditName();
  }

  File getFsTimeName() {
    StorageDirectory sd = null;
    // NameNodeFile.TIME shoul be same on all directories
    for (Iterator<StorageDirectory> it = 
             dirIterator(); it.hasNext();)
      sd = it.next();
    return getImageFile(sd, NameNodeFile.TIME);
  }

  /**
   * Return the name of the image file that is uploaded by periodic
   * checkpointing.
   */
  File[] getFsImageNameCheckpoint() {
    ArrayList<File> list = new ArrayList<File>();
    for (Iterator<StorageDirectory> it = 
                 dirIterator(NameNodeDirType.IMAGE); it.hasNext();) {
      list.add(getImageFile(it.next(), NameNodeFile.IMAGE_NEW));
    }
    return list.toArray(new File[list.size()]);
  }

  private boolean getDistributedUpgradeState() {
    FSNamesystem ns = getFSNamesystem();
    return ns == null ? false : ns.getDistributedUpgradeState();
  }

  private int getDistributedUpgradeVersion() {
    FSNamesystem ns = getFSNamesystem();
    return ns == null ? 0 : ns.getDistributedUpgradeVersion();
  }

  private void setDistributedUpgradeState(boolean uState, int uVersion) {
    getFSNamesystem().upgradeManager.setUpgradeState(uState, uVersion);
  }

  private void verifyDistributedUpgradeProgress(StartupOption startOpt
                                                ) throws IOException {
    if(startOpt == StartupOption.ROLLBACK || startOpt == StartupOption.IMPORT)
      return;
    UpgradeManager um = getFSNamesystem().upgradeManager;
    assert um != null : "FSNameSystem.upgradeManager is null.";
    if(startOpt != StartupOption.UPGRADE) {
      if(um.getUpgradeState())
        throw new IOException(
                    "\n   Previous distributed upgrade was not completed. "
                  + "\n   Please restart NameNode with -upgrade option.");
      if(um.getDistributedUpgrades() != null)
        throw new IOException("\n   Distributed upgrade for NameNode version " 
          + um.getUpgradeVersion() + " to current LV " + FSConstants.LAYOUT_VERSION
          + " is required.\n   Please restart NameNode with -upgrade option.");
    }
  }

  private void initializeDistributedUpgrade() throws IOException {
    UpgradeManagerNamenode um = getFSNamesystem().upgradeManager;
    if(! um.initializeUpgrade())
      return;
    // write new upgrade state into disk
    writeAll();
    NameNode.LOG.info("\n   Distributed upgrade for NameNode version " 
        + um.getUpgradeVersion() + " to current LV " 
        + FSConstants.LAYOUT_VERSION + " is initialized.");
  }

  /**
   * Retrieve checkpoint dirs from configuration.
   *  
   * @param conf the Configuration
   * @param defaultValue a default value for the attribute, if null
   * @return a Collection of URIs representing the values in 
   * dfs.namenode.checkpoint.dir configuration property
   */
  static Collection<URI> getCheckpointDirs(Configuration conf,
      String defaultValue) {
    Collection<String> dirNames = conf.getStringCollection(DFSConfigKeys.DFS_NAMENODE_CHECKPOINT_DIR_KEY);
    if (dirNames.size() == 0 && defaultValue != null) {
      dirNames.add(defaultValue);
    }
    return Util.stringCollectionAsURIs(dirNames);
  }

  static Collection<URI> getCheckpointEditsDirs(Configuration conf,
      String defaultName) {
    Collection<String> dirNames = 
      conf.getStringCollection(DFSConfigKeys.DFS_NAMENODE_CHECKPOINT_EDITS_DIR_KEY);
    if (dirNames.size() == 0 && defaultName != null) {
      dirNames.add(defaultName);
    }
    return Util.stringCollectionAsURIs(dirNames);
  }  
}

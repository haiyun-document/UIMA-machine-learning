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

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.EOFException;
import java.io.IOException;

import org.apache.hadoop.fs.Options.Rename;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.fs.permission.PermissionStatus;
import org.apache.hadoop.hdfs.protocol.Block;
import org.apache.hadoop.hdfs.protocol.DatanodeID;
import org.apache.hadoop.hdfs.protocol.FSConstants;
import org.apache.hadoop.hdfs.protocol.HdfsFileStatus;
import org.apache.hadoop.hdfs.protocol.LayoutVersion;
import org.apache.hadoop.hdfs.protocol.LayoutVersion.Feature;
import org.apache.hadoop.hdfs.security.token.delegation.DelegationTokenIdentifier;
import static org.apache.hadoop.hdfs.server.common.Util.now;
import org.apache.hadoop.hdfs.server.common.GenerationStamp;
import org.apache.hadoop.hdfs.server.common.Storage;
import org.apache.hadoop.hdfs.server.namenode.FSEditLog.Ops;
import org.apache.hadoop.io.BytesWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableFactories;
import org.apache.hadoop.io.WritableFactory;
import org.apache.hadoop.security.token.delegation.DelegationKey;

public class FSEditLogLoader {
  private final FSNamesystem fsNamesys;

  public FSEditLogLoader(FSNamesystem fsNamesys) {
    this.fsNamesys = fsNamesys;
  }
  
  /**
   * Load an edit log, and apply the changes to the in-memory structure
   * This is where we apply edits that we've been writing to disk all
   * along.
   */
  int loadFSEdits(EditLogInputStream edits) throws IOException {
    DataInputStream in = edits.getDataInputStream();
    long startTime = now();
    int numEdits = loadFSEdits(in, true);
    FSImage.LOG.info("Edits file " + edits.getName() 
        + " of size " + edits.length() + " edits # " + numEdits 
        + " loaded in " + (now()-startTime)/1000 + " seconds.");
    return numEdits;
  }

  int loadFSEdits(DataInputStream in, boolean closeOnExit) throws IOException {
    int numEdits = 0;
    int logVersion = 0;

    try {
      // Read log file version. Could be missing. 
      in.mark(4);
      // If edits log is greater than 2G, available method will return negative
      // numbers, so we avoid having to call available
      boolean available = true;
      try {
        logVersion = in.readByte();
      } catch (EOFException e) {
        available = false;
      }
      if (available) {
        in.reset();
        logVersion = in.readInt();
        if (logVersion < FSConstants.LAYOUT_VERSION) // future version
          throw new IOException(
                          "Unexpected version of the file system log file: "
                          + logVersion + ". Current version = " 
                          + FSConstants.LAYOUT_VERSION + ".");
      }
      assert logVersion <= Storage.LAST_UPGRADABLE_LAYOUT_VERSION :
                            "Unsupported version " + logVersion;
      numEdits = loadEditRecords(logVersion, in, false);
    } finally {
      if(closeOnExit)
        in.close();
    }
    if (logVersion != FSConstants.LAYOUT_VERSION) // other version
      numEdits++; // save this image asap
    return numEdits;
  }

  @SuppressWarnings("deprecation")
  int loadEditRecords(int logVersion, DataInputStream in,
      boolean closeOnExit) throws IOException {
    FSDirectory fsDir = fsNamesys.dir;
    int numEdits = 0;
    String clientName = null;
    String clientMachine = null;
    String path = null;
    int numOpAdd = 0, numOpClose = 0, numOpDelete = 0,
        numOpRenameOld = 0, numOpSetRepl = 0, numOpMkDir = 0,
        numOpSetPerm = 0, numOpSetOwner = 0, numOpSetGenStamp = 0,
        numOpTimes = 0, numOpRename = 0, numOpConcatDelete = 0, 
        numOpSymlink = 0, numOpGetDelegationToken = 0,
        numOpRenewDelegationToken = 0, numOpCancelDelegationToken = 0, 
        numOpUpdateMasterKey = 0, numOpOther = 0;

    try {
      while (true) {
        long timestamp = 0;
        long mtime = 0;
        long atime = 0;
        long blockSize = 0;
        byte opcode = -1;
        try {
          in.mark(1);
          opcode = in.readByte();
          if (opcode == Ops.OP_INVALID) {
            in.reset(); // reset back to end of file if somebody reads it again
            break; // no more transactions
          }
        } catch (EOFException e) {
          break; // no more transactions
        }
        numEdits++;
        switch (opcode) {
        case Ops.OP_ADD:
        case Ops.OP_CLOSE: {
          // versions > 0 support per file replication
          // get name and replication
          int length = in.readInt();
          if (-7 == logVersion && length != 3||
              -17 < logVersion && logVersion < -7 && length != 4 ||
              logVersion <= -17 && length != 5) {
              throw new IOException("Incorrect data format."  +
                                    " logVersion is " + logVersion +
                                    " but writables.length is " +
                                    length + ". ");
          }
          path = FSImageSerialization.readString(in);
          short replication = fsNamesys.adjustReplication(readShort(in));
          mtime = readLong(in);
          if (LayoutVersion.supports(Feature.FILE_ACCESS_TIME, logVersion)) {
            atime = readLong(in);
          }
          if (logVersion < -7) {
            blockSize = readLong(in);
          }
          // get blocks
          boolean isFileUnderConstruction = (opcode == Ops.OP_ADD);
          BlockInfo blocks[] = 
            readBlocks(in, logVersion, isFileUnderConstruction, replication);

          // Older versions of HDFS does not store the block size in inode.
          // If the file has more than one block, use the size of the
          // first block as the blocksize. Otherwise use the default
          // block size.
          if (-8 <= logVersion && blockSize == 0) {
            if (blocks.length > 1) {
              blockSize = blocks[0].getNumBytes();
            } else {
              long first = ((blocks.length == 1)? blocks[0].getNumBytes(): 0);
              blockSize = Math.max(fsNamesys.getDefaultBlockSize(), first);
            }
          }
           
          PermissionStatus permissions = fsNamesys.getUpgradePermission();
          if (logVersion <= -11) {
            permissions = PermissionStatus.read(in);
          }

          // clientname, clientMachine and block locations of last block.
          if (opcode == Ops.OP_ADD && logVersion <= -12) {
            clientName = FSImageSerialization.readString(in);
            clientMachine = FSImageSerialization.readString(in);
            if (-13 <= logVersion) {
              readDatanodeDescriptorArray(in);
            }
          } else {
            clientName = "";
            clientMachine = "";
          }

          // The open lease transaction re-creates a file if necessary.
          // Delete the file if it already exists.
          if (FSNamesystem.LOG.isDebugEnabled()) {
            FSNamesystem.LOG.debug(opcode + ": " + path + 
                                   " numblocks : " + blocks.length +
                                   " clientHolder " +  clientName +
                                   " clientMachine " + clientMachine);
          }

          fsDir.unprotectedDelete(path, mtime);

          // add to the file tree
          INodeFile node = (INodeFile)fsDir.unprotectedAddFile(
                                                    path, permissions,
                                                    blocks, replication, 
                                                    mtime, atime, blockSize);
          if (isFileUnderConstruction) {
            numOpAdd++;
            //
            // Replace current node with a INodeUnderConstruction.
            // Recreate in-memory lease record.
            //
            INodeFileUnderConstruction cons = new INodeFileUnderConstruction(
                                      node.getLocalNameBytes(),
                                      node.getReplication(), 
                                      node.getModificationTime(),
                                      node.getPreferredBlockSize(),
                                      node.getBlocks(),
                                      node.getPermissionStatus(),
                                      clientName, 
                                      clientMachine, 
                                      null);
            fsDir.replaceNode(path, node, cons);
            fsNamesys.leaseManager.addLease(cons.getClientName(), path);
          }
          break;
        } 
        case Ops.OP_SET_REPLICATION: {
          numOpSetRepl++;
          path = FSImageSerialization.readString(in);
          short replication = fsNamesys.adjustReplication(readShort(in));
          fsDir.unprotectedSetReplication(path, replication, null);
          break;
        } 
        case Ops.OP_CONCAT_DELETE: {
          numOpConcatDelete++;
          int length = in.readInt();
          if (length < 3) { // trg, srcs.., timestam
            throw new IOException("Incorrect data format. " 
                                  + "Mkdir operation.");
          }
          String trg = FSImageSerialization.readString(in);
          int srcSize = length - 1 - 1; //trg and timestamp
          String [] srcs = new String [srcSize];
          for(int i=0; i<srcSize;i++) {
            srcs[i]= FSImageSerialization.readString(in);
          }
          timestamp = readLong(in);
          fsDir.unprotectedConcat(trg, srcs);
          break;
        }
        case Ops.OP_RENAME_OLD: {
          numOpRenameOld++;
          int length = in.readInt();
          if (length != 3) {
            throw new IOException("Incorrect data format. " 
                                  + "Mkdir operation.");
          }
          String s = FSImageSerialization.readString(in);
          String d = FSImageSerialization.readString(in);
          timestamp = readLong(in);
          HdfsFileStatus dinfo = fsDir.getFileInfo(d, false);
          fsDir.unprotectedRenameTo(s, d, timestamp);
          fsNamesys.changeLease(s, d, dinfo);
          break;
        }
        case Ops.OP_DELETE: {
          numOpDelete++;
          int length = in.readInt();
          if (length != 2) {
            throw new IOException("Incorrect data format. " 
                                  + "delete operation.");
          }
          path = FSImageSerialization.readString(in);
          timestamp = readLong(in);
          fsDir.unprotectedDelete(path, timestamp);
          break;
        }
        case Ops.OP_MKDIR: {
          numOpMkDir++;
          PermissionStatus permissions = fsNamesys.getUpgradePermission();
          int length = in.readInt();
          if (-17 < logVersion && length != 2 ||
              logVersion <= -17 && length != 3) {
            throw new IOException("Incorrect data format. " 
                                  + "Mkdir operation.");
          }
          path = FSImageSerialization.readString(in);
          timestamp = readLong(in);

          // The disk format stores atimes for directories as well.
          // However, currently this is not being updated/used because of
          // performance reasons.
          if (LayoutVersion.supports(Feature.FILE_ACCESS_TIME, logVersion)) {
            atime = readLong(in);
          }

          if (logVersion <= -11) {
            permissions = PermissionStatus.read(in);
          }
          fsDir.unprotectedMkdir(path, permissions, timestamp);
          break;
        }
        case Ops.OP_SET_GENSTAMP: {
          numOpSetGenStamp++;
          long lw = in.readLong();
          fsNamesys.setGenerationStamp(lw);
          break;
        } 
        case Ops.OP_DATANODE_ADD: {
          numOpOther++;
          //Datanodes are not persistent any more.
          FSImageSerialization.DatanodeImage.skipOne(in);
          break;
        }
        case Ops.OP_DATANODE_REMOVE: {
          numOpOther++;
          DatanodeID nodeID = new DatanodeID();
          nodeID.readFields(in);
          //Datanodes are not persistent any more.
          break;
        }
        case Ops.OP_SET_PERMISSIONS: {
          numOpSetPerm++;
          fsDir.unprotectedSetPermission(
              FSImageSerialization.readString(in), FsPermission.read(in));
          break;
        }
        case Ops.OP_SET_OWNER: {
          numOpSetOwner++;
          fsDir.unprotectedSetOwner(FSImageSerialization.readString(in),
              FSImageSerialization.readString_EmptyAsNull(in),
              FSImageSerialization.readString_EmptyAsNull(in));
          break;
        }
        case Ops.OP_SET_NS_QUOTA: {
          fsDir.unprotectedSetQuota(FSImageSerialization.readString(in), 
                                    readLongWritable(in), 
                                    FSConstants.QUOTA_DONT_SET);
          break;
        }
        case Ops.OP_CLEAR_NS_QUOTA: {
          fsDir.unprotectedSetQuota(FSImageSerialization.readString(in),
                                    FSConstants.QUOTA_RESET,
                                    FSConstants.QUOTA_DONT_SET);
          break;
        }

        case Ops.OP_SET_QUOTA:
          fsDir.unprotectedSetQuota(FSImageSerialization.readString(in),
                                    readLongWritable(in),
                                    readLongWritable(in));
                                      
          break;

        case Ops.OP_TIMES: {
          numOpTimes++;
          int length = in.readInt();
          if (length != 3) {
            throw new IOException("Incorrect data format. " 
                                  + "times operation.");
          }
          path = FSImageSerialization.readString(in);
          mtime = readLong(in);
          atime = readLong(in);
          fsDir.unprotectedSetTimes(path, mtime, atime, true);
          break;
        }
        case Ops.OP_SYMLINK: {
          numOpSymlink++;
          int length = in.readInt();
          if (length != 4) {
            throw new IOException("Incorrect data format. " 
                                  + "symlink operation.");
          }
          path = FSImageSerialization.readString(in);
          String value = FSImageSerialization.readString(in);
          mtime = readLong(in);
          atime = readLong(in);
          PermissionStatus perm = PermissionStatus.read(in);
          fsDir.unprotectedSymlink(path, value, mtime, atime, perm);
          break;
        }
        case Ops.OP_RENAME: {
          numOpRename++;
          int length = in.readInt();
          if (length != 3) {
            throw new IOException("Incorrect data format. " 
                                  + "Mkdir operation.");
          }
          String s = FSImageSerialization.readString(in);
          String d = FSImageSerialization.readString(in);
          timestamp = readLong(in);
          Rename[] options = readRenameOptions(in);
          HdfsFileStatus dinfo = fsDir.getFileInfo(d, false);
          fsDir.unprotectedRenameTo(s, d, timestamp, options);
          fsNamesys.changeLease(s, d, dinfo);
          break;
        }
        case Ops.OP_GET_DELEGATION_TOKEN: {
          numOpGetDelegationToken++;
          DelegationTokenIdentifier delegationTokenId = 
              new DelegationTokenIdentifier();
          delegationTokenId.readFields(in);
          long expiryTime = readLong(in);
          fsNamesys.getDelegationTokenSecretManager()
              .addPersistedDelegationToken(delegationTokenId, expiryTime);
          break;
        }
        case Ops.OP_RENEW_DELEGATION_TOKEN: {
          numOpRenewDelegationToken++;
          DelegationTokenIdentifier delegationTokenId = 
              new DelegationTokenIdentifier();
          delegationTokenId.readFields(in);
          long expiryTime = readLong(in);
          fsNamesys.getDelegationTokenSecretManager()
              .updatePersistedTokenRenewal(delegationTokenId, expiryTime);
          break;
        }
        case Ops.OP_CANCEL_DELEGATION_TOKEN: {
          numOpCancelDelegationToken++;
          DelegationTokenIdentifier delegationTokenId = 
              new DelegationTokenIdentifier();
          delegationTokenId.readFields(in);
          fsNamesys.getDelegationTokenSecretManager()
              .updatePersistedTokenCancellation(delegationTokenId);
          break;
        }
        case Ops.OP_UPDATE_MASTER_KEY: {
          numOpUpdateMasterKey++;
          DelegationKey delegationKey = new DelegationKey();
          delegationKey.readFields(in);
          fsNamesys.getDelegationTokenSecretManager().updatePersistedMasterKey(
              delegationKey);
          break;
        }
        default: {
          throw new IOException("Never seen opcode " + opcode);
        }
        }
      }
    } catch (IOException ex) {
      check203UpgradeFailure(logVersion, ex);
    } finally {
      if(closeOnExit)
        in.close();
    }
    if (FSImage.LOG.isDebugEnabled()) {
      FSImage.LOG.debug("numOpAdd = " + numOpAdd + " numOpClose = " + numOpClose 
          + " numOpDelete = " + numOpDelete 
          + " numOpRenameOld = " + numOpRenameOld 
          + " numOpSetRepl = " + numOpSetRepl + " numOpMkDir = " + numOpMkDir
          + " numOpSetPerm = " + numOpSetPerm 
          + " numOpSetOwner = " + numOpSetOwner
          + " numOpSetGenStamp = " + numOpSetGenStamp 
          + " numOpTimes = " + numOpTimes
          + " numOpConcatDelete  = " + numOpConcatDelete
          + " numOpRename = " + numOpRename
          + " numOpGetDelegationToken = " + numOpGetDelegationToken
          + " numOpRenewDelegationToken = " + numOpRenewDelegationToken
          + " numOpCancelDelegationToken = " + numOpCancelDelegationToken
          + " numOpUpdateMasterKey = " + numOpUpdateMasterKey
          + " numOpOther = " + numOpOther);
    }
    return numEdits;
  }


  /**
   * A class to read in blocks stored in the old format. The only two
   * fields in the block were blockid and length.
   */
  static class BlockTwo implements Writable {
    long blkid;
    long len;

    static {                                      // register a ctor
      WritableFactories.setFactory
        (BlockTwo.class,
         new WritableFactory() {
           public Writable newInstance() { return new BlockTwo(); }
         });
    }


    BlockTwo() {
      blkid = 0;
      len = 0;
    }
    /////////////////////////////////////
    // Writable
    /////////////////////////////////////
    public void write(DataOutput out) throws IOException {
      out.writeLong(blkid);
      out.writeLong(len);
    }

    public void readFields(DataInput in) throws IOException {
      this.blkid = in.readLong();
      this.len = in.readLong();
    }
  }

  /** This method is defined for compatibility reason. */
  static private DatanodeDescriptor[] readDatanodeDescriptorArray(DataInput in
      ) throws IOException {
    DatanodeDescriptor[] locations = new DatanodeDescriptor[in.readInt()];
    for (int i = 0; i < locations.length; i++) {
      locations[i] = new DatanodeDescriptor();
      locations[i].readFieldsFromFSEditLog(in);
    }
    return locations;
  }

  static private short readShort(DataInputStream in) throws IOException {
    return Short.parseShort(FSImageSerialization.readString(in));
  }

  static private long readLong(DataInputStream in) throws IOException {
    return Long.parseLong(FSImageSerialization.readString(in));
  }
  
  // a place holder for reading a long
  private static final LongWritable longWritable = new LongWritable();

  /** Read an integer from an input stream */
  private static long readLongWritable(DataInputStream in) throws IOException {
    synchronized (longWritable) {
      longWritable.readFields(in);
      return longWritable.get();
    }
  }
  
  static Rename[] readRenameOptions(DataInputStream in) throws IOException {
    BytesWritable writable = new BytesWritable();
    writable.readFields(in);
    
    byte[] bytes = writable.getBytes();
    Rename[] options = new Rename[bytes.length];
    
    for (int i = 0; i < bytes.length; i++) {
      options[i] = Rename.valueOf(bytes[i]);
    }
    return options;
  }

  static private BlockInfo[] readBlocks(
      DataInputStream in,
      int logVersion,
      boolean isFileUnderConstruction,
      short replication) throws IOException {
    int numBlocks = in.readInt();
    BlockInfo[] blocks = new BlockInfo[numBlocks];
    Block blk = new Block();
    BlockTwo oldblk = new BlockTwo();
    for (int i = 0; i < numBlocks; i++) {
      if (logVersion <= -14) {
        blk.readFields(in);
      } else {
        oldblk.readFields(in);
        blk.set(oldblk.blkid, oldblk.len,
                GenerationStamp.GRANDFATHER_GENERATION_STAMP);
      }
      if(isFileUnderConstruction && i == numBlocks-1)
        blocks[i] = new BlockInfoUnderConstruction(blk, replication);
      else
        blocks[i] = new BlockInfo(blk, replication);
    }
    return blocks;
  }
  
  /** 
   * Throw appropriate exception during upgrade from 203, when editlog loading
   * could fail due to opcode conflicts.
   */
  private void check203UpgradeFailure(int logVersion, IOException ex)
      throws IOException {
    // 0.20.203 version version has conflicting opcodes with the later releases.
    // The editlog must be emptied by restarting the namenode, before proceeding
    // with the upgrade.
    if (Storage.is203LayoutVersion(logVersion)
        && logVersion != FSConstants.LAYOUT_VERSION) {
      String msg = "During upgrade failed to load the editlog version "
          + logVersion + " from release 0.20.203. Please go back to the old "
          + " release and restart the namenode. This empties the editlog "
          + " and saves the namespace. Resume the upgrade after this step.";
      throw new IOException(msg, ex);
    } else {
      throw ex;
    }
  }
}

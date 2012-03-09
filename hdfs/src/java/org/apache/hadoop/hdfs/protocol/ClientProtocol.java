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
package org.apache.hadoop.hdfs.protocol;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.avro.reflect.Nullable;

import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;
import org.apache.hadoop.fs.ContentSummary;
import org.apache.hadoop.fs.CreateFlag;
import org.apache.hadoop.fs.FsServerDefaults;
import org.apache.hadoop.fs.Options;
import org.apache.hadoop.fs.FileAlreadyExistsException;
import org.apache.hadoop.fs.ParentNotDirectoryException;
import org.apache.hadoop.fs.InvalidPathException;
import org.apache.hadoop.fs.UnresolvedLinkException;
import org.apache.hadoop.fs.Options.Rename;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.hdfs.DFSConfigKeys;
import org.apache.hadoop.hdfs.protocol.FSConstants.UpgradeAction;
import org.apache.hadoop.hdfs.server.common.UpgradeStatusReport;
import org.apache.hadoop.hdfs.server.namenode.NotReplicatedYetException;
import org.apache.hadoop.hdfs.server.namenode.SafeModeException;
import org.apache.hadoop.io.EnumSetWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.ipc.VersionedProtocol;
import org.apache.hadoop.security.AccessControlException;
import org.apache.hadoop.security.KerberosInfo;
import org.apache.hadoop.security.token.Token;
import org.apache.hadoop.security.token.TokenInfo;
import org.apache.hadoop.hdfs.security.token.delegation.DelegationTokenIdentifier;
import org.apache.hadoop.hdfs.security.token.delegation.DelegationTokenSelector;

/**********************************************************************
 * ClientProtocol is used by user code via 
 * {@link org.apache.hadoop.hdfs.DistributedFileSystem} class to communicate 
 * with the NameNode.  User code can manipulate the directory namespace, 
 * as well as open/close file streams, etc.
 *
 **********************************************************************/
@InterfaceAudience.Private
@InterfaceStability.Evolving
@KerberosInfo(
    serverPrincipal = DFSConfigKeys.DFS_NAMENODE_USER_NAME_KEY)
@TokenInfo(DelegationTokenSelector.class)
public interface ClientProtocol extends VersionedProtocol {

  /**
   * Compared to the previous version the following changes have been introduced:
   * (Only the latest change is reflected.
   * The log of historical changes can be retrieved from the svn).
   * 65: recoverLease return if the file is closed or not
   */
  public static final long versionID = 65L;
  
  ///////////////////////////////////////
  // File contents
  ///////////////////////////////////////
  /**
   * Get locations of the blocks of the specified file within the specified range.
   * DataNode locations for each block are sorted by
   * the proximity to the client.
   * <p>
   * Return {@link LocatedBlocks} which contains
   * file length, blocks and their locations.
   * DataNode locations for each block are sorted by
   * the distance to the client's address.
   * <p>
   * The client will then have to contact 
   * one of the indicated DataNodes to obtain the actual data.
   * 
   * @param src file name
   * @param offset range start offset
   * @param length range length
   *
   * @return file length and array of blocks with their locations
   *
   * @throws AccessControlException If access is denied
   * @throws FileNotFoundException If file <code>src</code> does not exist
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred
   */
  @Nullable
  public LocatedBlocks getBlockLocations(String src,
                                         long offset,
                                         long length) 
      throws AccessControlException, FileNotFoundException,
      UnresolvedLinkException, IOException;

  /**
   * Get server default values for a number of configuration params.
   * @return a set of server default configuration values
   * @throws IOException
   */
  public FsServerDefaults getServerDefaults() throws IOException;

  /**
   * Create a new file entry in the namespace.
   * <p>
   * This will create an empty file specified by the source path.
   * The path should reflect a full path originated at the root.
   * The name-node does not have a notion of "current" directory for a client.
   * <p>
   * Once created, the file is visible and available for read to other clients.
   * Although, other clients cannot {@link #delete(String, boolean)}, re-create or 
   * {@link #rename(String, String)} it until the file is completed
   * or explicitly as a result of lease expiration.
   * <p>
   * Blocks have a maximum size.  Clients that intend to create
   * multi-block files must also use 
   * {@link #addBlock(String, String, Block, DatanodeInfo[])}
   *
   * @param src path of the file being created.
   * @param masked masked permission.
   * @param clientName name of the current client.
   * @param flag indicates whether the file should be 
   * overwritten if it already exists or create if it does not exist or append.
   * @param createParent create missing parent directory if true
   * @param replication block replication factor.
   * @param blockSize maximum block size.
   * 
   * @throws AccessControlException If access is denied
   * @throws AlreadyBeingCreatedException if the path does not exist.
   * @throws DSQuotaExceededException If file creation violates disk space 
   *           quota restriction
   * @throws FileAlreadyExistsException If file <code>src</code> already exists
   * @throws FileNotFoundException If parent of <code>src</code> does not exist
   *           and <code>createParent</code> is false
   * @throws ParentNotDirectoryException If parent of <code>src</code> is not a
   *           directory.
   * @throws NSQuotaExceededException If file creation violates name space 
   *           quota restriction
   * @throws SafeModeException create not allowed in safemode
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred
   *
   * RuntimeExceptions:
   * @throws InvalidPathException Path <code>src</code> is invalid
   */
  public void create(String src, FsPermission masked, String clientName,
      EnumSetWritable<CreateFlag> flag, boolean createParent,
      short replication, long blockSize) throws AccessControlException,
      AlreadyBeingCreatedException, DSQuotaExceededException,
      FileAlreadyExistsException, FileNotFoundException,
      NSQuotaExceededException, ParentNotDirectoryException, SafeModeException,
      UnresolvedLinkException, IOException;

  /**
   * Append to the end of the file. 
   * @param src path of the file being created.
   * @param clientName name of the current client.
   * @return information about the last partial block if any.
   * @throws AccessControlException if permission to append file is 
   * denied by the system. As usually on the client side the exception will 
   * be wrapped into {@link org.apache.hadoop.ipc.RemoteException}.
   * Allows appending to an existing file if the server is
   * configured with the parameter dfs.support.append set to true, otherwise
   * throws an IOException.
   * 
   * @throws AccessControlException If permission to append to file is denied
   * @throws FileNotFoundException If file <code>src</code> is not found
   * @throws DSQuotaExceededException If append violates disk space quota 
   *           restriction
   * @throws SafeModeException append not allowed in safemode
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred.
   *
   * RuntimeExceptions:
   * @throws UnsupportedOperationException if append is not supported
   */
  public LocatedBlock append(String src, String clientName)
      throws AccessControlException, DSQuotaExceededException,
      FileNotFoundException, SafeModeException, UnresolvedLinkException,
      IOException;

  /**
   * Set replication for an existing file.
   * <p>
   * The NameNode sets replication to the new value and returns.
   * The actual block replication is not expected to be performed during  
   * this method call. The blocks will be populated or removed in the 
   * background as the result of the routine block maintenance procedures.
   * 
   * @param src file name
   * @param replication new replication
   * 
   * @return true if successful;
   *         false if file does not exist or is a directory
   *
   * @throws AccessControlException If access is denied
   * @throws DSQuotaExceededException If replication violates disk space 
   *           quota restriction
   * @throws FileNotFoundException If file <code>src</code> is not found
   * @throws SafeModeException not allowed in safemode
   * @throws UnresolvedLinkException if <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred
   */
  public boolean setReplication(String src, short replication)
      throws AccessControlException, DSQuotaExceededException,
      FileNotFoundException, SafeModeException, UnresolvedLinkException,
      IOException;

  /**
   * Set permissions for an existing file/directory.
   * 
   * @throws AccessControlException If access is denied
   * @throws FileNotFoundException If file <code>src</code> is not found
   * @throws SafeModeException not allowed in safemode
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred
   */
  public void setPermission(String src, FsPermission permission)
      throws AccessControlException, FileNotFoundException, SafeModeException,
      UnresolvedLinkException, IOException;

  /**
   * Set Owner of a path (i.e. a file or a directory).
   * The parameters username and groupname cannot both be null.
   * @param src
   * @param username If it is null, the original username remains unchanged.
   * @param groupname If it is null, the original groupname remains unchanged.
   *
   * @throws AccessControlException If access is denied
   * @throws FileNotFoundException If file <code>src</code> is not found
   * @throws SafeModeException not allowed in safemode
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred
   */
  public void setOwner(String src, String username, String groupname)
      throws AccessControlException, FileNotFoundException, SafeModeException,
      UnresolvedLinkException, IOException;

  /**
   * The client can give up on a blcok by calling abandonBlock().
   * The client can then
   * either obtain a new block, or complete or abandon the file.
   * Any partial writes to the block will be discarded.
   * 
   * @throws AccessControlException If access is denied
   * @throws FileNotFoundException file <code>src</code> is not found
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred
   */
  public void abandonBlock(Block b, String src, String holder)
      throws AccessControlException, FileNotFoundException,
      UnresolvedLinkException, IOException;

  /**
   * A client that wants to write an additional block to the 
   * indicated filename (which must currently be open for writing)
   * should call addBlock().  
   *
   * addBlock() allocates a new block and datanodes the block data
   * should be replicated to.
   * 
   * addBlock() also commits the previous block by reporting
   * to the name-node the actual generation stamp and the length
   * of the block that the client has transmitted to data-nodes.
   *
   * @param src the file being created
   * @param clientName the name of the client that adds the block
   * @param previous  previous block
   * @param excludeNodes a list of nodes that should not be
   * allocated for the current block
   *
   * @return LocatedBlock allocated block information.
   *
   * @throws AccessControlException If access is denied
   * @throws FileNotFoundException If file <code>src</code> is not found
   * @throws NotReplicatedYetException previous blocks of the file are not
   *           replicated yet. Blocks cannot be added until replication
   *           completes.
   * @throws SafeModeException create not allowed in safemode
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred
   */
  public LocatedBlock addBlock(String src, String clientName,
      @Nullable Block previous, @Nullable DatanodeInfo[] excludeNodes)
      throws AccessControlException, FileNotFoundException,
      NotReplicatedYetException, SafeModeException, UnresolvedLinkException,
      IOException;

  /**
   * The client is done writing data to the given filename, and would 
   * like to complete it.  
   *
   * The function returns whether the file has been closed successfully.
   * If the function returns false, the caller should try again.
   * 
   * close() also commits the last block of the file by reporting
   * to the name-node the actual generation stamp and the length
   * of the block that the client has transmitted to data-nodes.
   *
   * A call to complete() will not return true until all the file's
   * blocks have been replicated the minimum number of times.  Thus,
   * DataNode failures may cause a client to call complete() several
   * times before succeeding.
   *
   * @throws AccessControlException If access is denied
   * @throws FileNotFoundException If file <code>src</code> is not found
   * @throws SafeModeException create not allowed in safemode
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink 
   * @throws IOException If an I/O error occurred
   */
  public boolean complete(String src, String clientName, Block last)
      throws AccessControlException, FileNotFoundException, SafeModeException,
      UnresolvedLinkException, IOException;

  /**
   * The client wants to report corrupted blocks (blocks with specified
   * locations on datanodes).
   * @param blocks Array of located blocks to report
   */
  public void reportBadBlocks(LocatedBlock[] blocks) throws IOException;

  ///////////////////////////////////////
  // Namespace management
  ///////////////////////////////////////
  /**
   * Rename an item in the file system namespace.
   * @param src existing file or directory name.
   * @param dst new name.
   * @return true if successful, or false if the old name does not exist
   * or if the new name already belongs to the namespace.
   * 
   * @throws IOException an I/O error occurred
   * 
   * @deprecated Use {@link #rename(String, String, Options.Rename...)} instead.
   */
  @Deprecated
  public boolean rename(String src, String dst) 
      throws UnresolvedLinkException, IOException;

  /**
   * Moves blocks from srcs to trg and delete srcs
   * 
   * @param trg existing file
   * @param srcs - list of existing files (same block size, same replication)
   * @throws IOException if some arguments are invalid
   * @throws UnresolvedLinkException if <code>trg</code> or <code>srcs</code>
   *           contains a symlink
   */
  public void concat(String trg, String[] srcs) 
      throws IOException, UnresolvedLinkException;

  /**
   * Rename src to dst.
   * <ul>
   * <li>Fails if src is a file and dst is a directory.
   * <li>Fails if src is a directory and dst is a file.
   * <li>Fails if the parent of dst does not exist or is a file.
   * </ul>
   * <p>
   * Without OVERWRITE option, rename fails if the dst already exists.
   * With OVERWRITE option, rename overwrites the dst, if it is a file 
   * or an empty directory. Rename fails if dst is a non-empty directory.
   * <p>
   * This implementation of rename is atomic.
   * <p>
   * @param src existing file or directory name.
   * @param dst new name.
   * @param options Rename options
   * 
   * @throws AccessControlException If access is denied
   * @throws DSQuotaExceededException If rename violates disk space 
   *           quota restriction
   * @throws FileAlreadyExistsException If <code>dst</code> already exists and
   *           <code>options</options> has {@link Rename#OVERWRITE} option
   *           false.
   * @throws FileNotFoundException If <code>src</code> does not exist
   * @throws NSQuotaExceededException If rename violates namespace 
   *           quota restriction
   * @throws ParentNotDirectoryException If parent of <code>dst</code> 
   *           is not a directory
   * @throws SafeModeException rename not allowed in safemode
   * @throws UnresolvedLinkException If <code>src</code> or
   *           <code>dst</code> contains a symlink
   * @throws IOException If an I/O error occurred
   */
  public void rename(String src, String dst, Options.Rename... options)
      throws AccessControlException, DSQuotaExceededException,
      FileAlreadyExistsException, FileNotFoundException,
      NSQuotaExceededException, ParentNotDirectoryException, SafeModeException,
      UnresolvedLinkException, IOException;
  
  /**
   * Delete the given file or directory from the file system.
   * <p>
   * Any blocks belonging to the deleted files will be garbage-collected.
   * 
   * @param src existing name.
   * @return true only if the existing file or directory was actually removed 
   * from the file system. 
   * @throws UnresolvedLinkException if <code>src</code> contains a symlink. 
   * @deprecated use {@link #delete(String, boolean)} istead.
   */
  @Deprecated
  public boolean delete(String src) 
      throws IOException, UnresolvedLinkException;

  /**
   * Delete the given file or directory from the file system.
   * <p>
   * same as delete but provides a way to avoid accidentally 
   * deleting non empty directories programmatically. 
   * @param src existing name
   * @param recursive if true deletes a non empty directory recursively,
   * else throws an exception.
   * @return true only if the existing file or directory was actually removed 
   * from the file system.
   * 
   * @throws AccessControlException If access is denied
   * @throws FileNotFoundException If file <code>src</code> is not found
   * @throws SafeModeException create not allowed in safemode
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred
   */
  public boolean delete(String src, boolean recursive)
      throws AccessControlException, FileNotFoundException, SafeModeException,
      UnresolvedLinkException, IOException;
  
  /**
   * Create a directory (or hierarchy of directories) with the given
   * name and permission.
   *
   * @param src The path of the directory being created
   * @param masked The masked permission of the directory being created
   * @param createParent create missing parent directory if true
   *
   * @return True if the operation success.
   *
   * @throws AccessControlException If access is denied
   * @throws FileAlreadyExistsException If <code>src</code> already exists
   * @throws FileNotFoundException If parent of <code>src</code> does not exist
   *           and <code>createParent</code> is false
   * @throws NSQuotaExceededException If file creation violates quota restriction
   * @throws ParentNotDirectoryException If parent of <code>src</code> 
   *           is not a directory
   * @throws SafeModeException create not allowed in safemode
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred.
   *
   * RunTimeExceptions:
   * @throws InvalidPathException If <code>src</code> is invalid
   */
  public boolean mkdirs(String src, FsPermission masked, boolean createParent)
      throws AccessControlException, FileAlreadyExistsException,
      FileNotFoundException, NSQuotaExceededException,
      ParentNotDirectoryException, SafeModeException, UnresolvedLinkException,
      IOException;

  /**
   * Get a partial listing of the indicated directory
   *
   * @param src the directory name
   * @param startAfter the name to start listing after encoded in java UTF8
   * @param needLocation if the FileStatus should contain block locations
   *
   * @return a partial listing starting after startAfter
   *
   * @throws AccessControlException permission denied
   * @throws FileNotFoundException file <code>src</code> is not found
   * @throws UnresolvedLinkException If <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred
   */
  public DirectoryListing getListing(String src,
                                     byte[] startAfter,
                                     boolean needLocation)
      throws AccessControlException, FileNotFoundException,
      UnresolvedLinkException, IOException;

  ///////////////////////////////////////
  // System issues and management
  ///////////////////////////////////////

  /**
   * Client programs can cause stateful changes in the NameNode
   * that affect other clients.  A client may obtain a file and 
   * neither abandon nor complete it.  A client might hold a series
   * of locks that prevent other clients from proceeding.
   * Clearly, it would be bad if a client held a bunch of locks
   * that it never gave up.  This can happen easily if the client
   * dies unexpectedly.
   * <p>
   * So, the NameNode will revoke the locks and live file-creates
   * for clients that it thinks have died.  A client tells the
   * NameNode that it is still alive by periodically calling
   * renewLease().  If a certain amount of time passes since
   * the last call to renewLease(), the NameNode assumes the
   * client has died.
   *
   * @throws AccessControlException permission denied
   * @throws IOException If an I/O error occurred
   */
  public void renewLease(String clientName) throws AccessControlException,
      IOException;

  /**
   * Start lease recovery.
   * Lightweight NameNode operation to trigger lease recovery
   * 
   * @param src path of the file to start lease recovery
   * @param clientName name of the current client
   * @return true if the file is already closed
   * @throws IOException
   */
  public boolean recoverLease(String src, String clientName) throws IOException;

  public int GET_STATS_CAPACITY_IDX = 0;
  public int GET_STATS_USED_IDX = 1;
  public int GET_STATS_REMAINING_IDX = 2;
  public int GET_STATS_UNDER_REPLICATED_IDX = 3;
  public int GET_STATS_CORRUPT_BLOCKS_IDX = 4;
  public int GET_STATS_MISSING_BLOCKS_IDX = 5;
  
  /**
   * Get a set of statistics about the filesystem.
   * Right now, only three values are returned.
   * <ul>
   * <li> [0] contains the total storage capacity of the system, in bytes.</li>
   * <li> [1] contains the total used space of the system, in bytes.</li>
   * <li> [2] contains the available storage of the system, in bytes.</li>
   * <li> [3] contains number of under replicated blocks in the system.</li>
   * <li> [4] contains number of blocks with a corrupt replica. </li>
   * <li> [5] contains number of blocks without any good replicas left. </li>
   * </ul>
   * Use public constants like {@link #GET_STATS_CAPACITY_IDX} in place of 
   * actual numbers to index into the array.
   */
  public long[] getStats() throws IOException;

  /**
   * Get a report on the system's current datanodes.
   * One DatanodeInfo object is returned for each DataNode.
   * Return live datanodes if type is LIVE; dead datanodes if type is DEAD;
   * otherwise all datanodes if type is ALL.
   */
  public DatanodeInfo[] getDatanodeReport(FSConstants.DatanodeReportType type)
      throws IOException;

  /**
   * Get the block size for the given file.
   * @param filename The name of the file
   * @return The number of bytes in each block
   * @throws IOException
   * @throws UnresolvedLinkException if the path contains a symlink. 
   */
  public long getPreferredBlockSize(String filename) 
      throws IOException, UnresolvedLinkException;

  /**
   * Enter, leave or get safe mode.
   * <p>
   * Safe mode is a name node state when it
   * <ol><li>does not accept changes to name space (read-only), and</li>
   * <li>does not replicate or delete blocks.</li></ol>
   * 
   * <p>
   * Safe mode is entered automatically at name node startup.
   * Safe mode can also be entered manually using
   * {@link #setSafeMode(FSConstants.SafeModeAction) setSafeMode(SafeModeAction.SAFEMODE_GET)}.
   * <p>
   * At startup the name node accepts data node reports collecting
   * information about block locations.
   * In order to leave safe mode it needs to collect a configurable
   * percentage called threshold of blocks, which satisfy the minimal 
   * replication condition.
   * The minimal replication condition is that each block must have at least
   * <tt>dfs.namenode.replication.min</tt> replicas.
   * When the threshold is reached the name node extends safe mode
   * for a configurable amount of time
   * to let the remaining data nodes to check in before it
   * will start replicating missing blocks.
   * Then the name node leaves safe mode.
   * <p>
   * If safe mode is turned on manually using
   * {@link #setSafeMode(FSConstants.SafeModeAction) setSafeMode(SafeModeAction.SAFEMODE_ENTER)}
   * then the name node stays in safe mode until it is manually turned off
   * using {@link #setSafeMode(FSConstants.SafeModeAction) setSafeMode(SafeModeAction.SAFEMODE_LEAVE)}.
   * Current state of the name node can be verified using
   * {@link #setSafeMode(FSConstants.SafeModeAction) setSafeMode(SafeModeAction.SAFEMODE_GET)}
   * <h4>Configuration parameters:</h4>
   * <tt>dfs.safemode.threshold.pct</tt> is the threshold parameter.<br>
   * <tt>dfs.safemode.extension</tt> is the safe mode extension parameter.<br>
   * <tt>dfs.namenode.replication.min</tt> is the minimal replication parameter.
   * 
   * <h4>Special cases:</h4>
   * The name node does not enter safe mode at startup if the threshold is 
   * set to 0 or if the name space is empty.<br>
   * If the threshold is set to 1 then all blocks need to have at least 
   * minimal replication.<br>
   * If the threshold value is greater than 1 then the name node will not be 
   * able to turn off safe mode automatically.<br>
   * Safe mode can always be turned off manually.
   * 
   * @param action  <ul> <li>0 leave safe mode;</li>
   *                <li>1 enter safe mode;</li>
   *                <li>2 get safe mode state.</li></ul>
   * @return <ul><li>0 if the safe mode is OFF or</li> 
   *         <li>1 if the safe mode is ON.</li></ul>
   *                   
   * @throws IOException
   */
  public boolean setSafeMode(FSConstants.SafeModeAction action) 
      throws IOException;

  /**
   * Save namespace image.
   * <p>
   * Saves current namespace into storage directories and reset edits log.
   * Requires superuser privilege and safe mode.
   * 
   * @throws AccessControlException if the superuser privilege is violated.
   * @throws IOException if image creation failed.
   */
  public void saveNamespace() throws AccessControlException, IOException;

  /**
   * Enable/Disable restore failed storage.
   * <p>
   * sets flag to enable restore of failed storage replicas
   * 
   * @throws AccessControlException if the superuser privilege is violated.
   */
  public boolean restoreFailedStorage(String arg) throws AccessControlException;

  /**
   * Tells the namenode to reread the hosts and exclude files. 
   * @throws IOException
   */
  public void refreshNodes() throws IOException;

  /**
   * Finalize previous upgrade.
   * Remove file system state saved during the upgrade.
   * The upgrade will become irreversible.
   * 
   * @throws IOException
   */
  public void finalizeUpgrade() throws IOException;

  /**
   * Report distributed upgrade progress or force current upgrade to proceed.
   * 
   * @param action {@link FSConstants.UpgradeAction} to perform
   * @return upgrade status information or null if no upgrades are in progress
   * @throws IOException
   */
  @Nullable
  public UpgradeStatusReport distributedUpgradeProgress(UpgradeAction action) 
      throws IOException;

  /**
   * Dumps namenode data structures into specified file. If the file
   * already exists, then append.
   *
   * @throws IOException
   */
  public void metaSave(String filename) throws IOException;
  
  /**
   * Get the file info for a specific file or directory.
   * @param src The string representation of the path to the file
   *
   * @return object containing information regarding the file
   *         or null if file not found
   * @throws AccessControlException permission denied
   * @throws FileNotFoundException file <code>src</code> is not found
   * @throws UnresolvedLinkException if the path contains a symlink. 
   * @throws IOException If an I/O error occurred        
   */
  @Nullable
  public HdfsFileStatus getFileInfo(String src) throws AccessControlException,
      FileNotFoundException, UnresolvedLinkException, IOException;

  /**
   * Get the file info for a specific file or directory. If the path 
   * refers to a symlink then the FileStatus of the symlink is returned.
   * @param src The string representation of the path to the file
   *
   * @return object containing information regarding the file
   *         or null if file not found
   *
   * @throws AccessControlException permission denied
   * @throws UnresolvedLinkException if <code>src</code> contains a symlink
   * @throws IOException If an I/O error occurred        
   */
  public HdfsFileStatus getFileLinkInfo(String src)
      throws AccessControlException, UnresolvedLinkException, IOException;
  
  /**
   * Get {@link ContentSummary} rooted at the specified directory.
   * @param path The string representation of the path
   *
   * @throws AccessControlException permission denied
   * @throws FileNotFoundException file <code>path</code> is not found
   * @throws UnresolvedLinkException if <code>path</code> contains a symlink. 
   * @throws IOException If an I/O error occurred
   */
  public ContentSummary getContentSummary(String path)
      throws AccessControlException, FileNotFoundException,
      UnresolvedLinkException, IOException;

  /**
   * Set the quota for a directory.
   * @param path  The string representation of the path to the directory
   * @param namespaceQuota Limit on the number of names in the tree rooted 
   *                       at the directory
   * @param diskspaceQuota Limit on disk space occupied all the files under
   *                       this directory. 
   * <br><br>
   *                       
   * The quota can have three types of values : (1) 0 or more will set 
   * the quota to that value, (2) {@link FSConstants#QUOTA_DONT_SET}  implies 
   * the quota will not be changed, and (3) {@link FSConstants#QUOTA_RESET} 
   * implies the quota will be reset. Any other value is a runtime error.
   * 
   * @throws AccessControlException permission denied
   * @throws FileNotFoundException file <code>path</code> is not found
   * @throws QuotaExceededException if the directory size 
   *           is greater than the given quota
   * @throws UnresolvedLinkException if the <code>path</code> contains a symlink. 
   * @throws IOException If an I/O error occurred
   */
  public void setQuota(String path, long namespaceQuota, long diskspaceQuota)
      throws AccessControlException, FileNotFoundException,
      UnresolvedLinkException, IOException;

  /**
   * Write all metadata for this file into persistent storage.
   * The file must be currently open for writing.
   * @param src The string representation of the path
   * @param client The string representation of the client
   * 
   * @throws AccessControlException permission denied
   * @throws FileNotFoundException file <code>src</code> is not found
   * @throws UnresolvedLinkException if <code>src</code> contains a symlink. 
   * @throws IOException If an I/O error occurred
   */
  public void fsync(String src, String client) 
      throws AccessControlException, FileNotFoundException, 
      UnresolvedLinkException, IOException;

  /**
   * Sets the modification and access time of the file to the specified time.
   * @param src The string representation of the path
   * @param mtime The number of milliseconds since Jan 1, 1970.
   *              Setting mtime to -1 means that modification time should not be set
   *              by this call.
   * @param atime The number of milliseconds since Jan 1, 1970.
   *              Setting atime to -1 means that access time should not be set
   *              by this call.
   *              
   * @throws AccessControlException permission denied
   * @throws FileNotFoundException file <code>src</code> is not found
   * @throws UnresolvedLinkException if <code>src</code> contains a symlink. 
   * @throws IOException If an I/O error occurred
   */
  public void setTimes(String src, long mtime, long atime)
      throws AccessControlException, FileNotFoundException, 
      UnresolvedLinkException, IOException;

  /**
   * Create symlink to a file or directory.
   * @param target The path of the destination that the
   *               link points to.
   * @param link The path of the link being created.
   * @param dirPerm permissions to use when creating parent directories
   * @param createParent - if true then missing parent dirs are created
   *                       if false then parent must exist
   *
   * @throws AccessControlException permission denied
   * @throws FileAlreadyExistsException If file <code>link</code> already exists
   * @throws FileNotFoundException If parent of <code>link</code> does not exist
   *           and <code>createParent</code> is false
   * @throws ParentNotDirectoryException If parent of <code>link</code> is not a
   *           directory.
   * @throws UnresolvedLinkException if <code>link</target> contains a symlink. 
   * @throws IOException If an I/O error occurred
   */
  public void createSymlink(String target, String link, FsPermission dirPerm,
      boolean createParent) throws AccessControlException,
      FileAlreadyExistsException, FileNotFoundException,
      ParentNotDirectoryException, SafeModeException, UnresolvedLinkException,
      IOException;

  /**
   * Return the target of the given symlink. If there is an intermediate
   * symlink in the path (ie a symlink leading up to the final path component)
   * then the given path is returned with this symlink resolved.
   *
   * @param path The path with a link that needs resolution.
   * @return The path after resolving the first symbolic link in the path.
   * @throws AccessControlException permission denied
   * @throws FileNotFoundException If <code>path</code> does not exist
   * @throws IOException If the given path does not refer to a symlink
   *           or an I/O error occurred
   */
  public String getLinkTarget(String path) throws AccessControlException,
      FileNotFoundException, IOException; 
  
  /**
   * Get a new generation stamp together with an access token for 
   * a block under construction
   * 
   * This method is called only when a client needs to recover a failed
   * pipeline or set up a pipeline for appending to a block.
   * 
   * @param block a block
   * @param clientName the name of the client
   * @return a located block with a new generation stamp and an access token
   * @throws IOException if any error occurs
   */
  public LocatedBlock updateBlockForPipeline(Block block, String clientName) 
      throws IOException;

  /**
   * Update a pipeline for a block under construction
   * 
   * @param clientName the name of the client
   * @param oldBlock the old block
   * @param newBlock the new block containing new generation stamp and length
   * @param newNodes datanodes in the pipeline
   * @throws IOException if any error occurs
   */
  public void updatePipeline(String clientName, Block oldBlock, 
      Block newBlock, DatanodeID[] newNodes)
      throws IOException;

  /**
   * Get a valid Delegation Token.
   * 
   * @param renewer the designated renewer for the token
   * @return Token<DelegationTokenIdentifier>
   * @throws IOException
   */
  public Token<DelegationTokenIdentifier> getDelegationToken(Text renewer) 
      throws IOException;

  /**
   * Renew an existing delegation token.
   * 
   * @param token delegation token obtained earlier
   * @return the new expiration time
   * @throws IOException
   */
  public long renewDelegationToken(Token<DelegationTokenIdentifier> token)
      throws IOException;
  
  /**
   * Cancel an existing delegation token.
   * 
   * @param token delegation token
   * @throws IOException
   */
  public void cancelDelegationToken(Token<DelegationTokenIdentifier> token)
      throws IOException;
}

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

import static org.apache.hadoop.hdfs.server.common.Util.fileAsURI;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.channels.FileChannel;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.CommonConfigurationKeys;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.hdfs.protocol.Block;
import org.apache.hadoop.hdfs.protocol.BlockListAsLongs;
import org.apache.hadoop.hdfs.protocol.ClientProtocol;
import org.apache.hadoop.hdfs.protocol.DatanodeInfo;
import org.apache.hadoop.hdfs.protocol.FSConstants.DatanodeReportType;
import org.apache.hadoop.hdfs.server.common.HdfsConstants.StartupOption;
import org.apache.hadoop.hdfs.server.datanode.DataNode;
import org.apache.hadoop.hdfs.server.datanode.FSDatasetInterface;
import org.apache.hadoop.hdfs.server.datanode.SimulatedFSDataset;
import org.apache.hadoop.hdfs.server.namenode.FSNamesystem;
import org.apache.hadoop.hdfs.server.namenode.NameNode;
import org.apache.hadoop.hdfs.server.namenode.NameNodeAdapter;
import org.apache.hadoop.hdfs.server.protocol.DatanodeProtocol;
import org.apache.hadoop.hdfs.server.protocol.NamenodeProtocol;
import org.apache.hadoop.hdfs.server.protocol.NamenodeProtocols;
import org.apache.hadoop.hdfs.tools.DFSAdmin;
import org.apache.hadoop.net.DNSToSwitchMapping;
import org.apache.hadoop.net.NetUtils;
import org.apache.hadoop.net.StaticMapping;
import org.apache.hadoop.security.RefreshUserMappingsProtocol;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.security.authorize.ProxyUsers;
import org.apache.hadoop.security.authorize.RefreshAuthorizationPolicyProtocol;
import org.apache.hadoop.util.StringUtils;
import org.apache.hadoop.util.ToolRunner;

/**
 * This class creates a single-process DFS cluster for junit testing.
 * The data directories for non-simulated DFS are under the testing directory.
 * For simulated data nodes, no underlying fs storage is used.
 */
public class MiniDFSCluster {

  /**
   * Class to construct instances of MiniDFSClusters with specific options.
   */
  public static class Builder {
    private int nameNodePort = 0;
    private final Configuration conf;
    private int numDataNodes = 1;
    private boolean format = true;
    private boolean manageNameDfsDirs = true;
    private boolean manageDataDfsDirs = true;
    private StartupOption option = null;
    private String[] racks = null; 
    private String [] hosts = null;
    private long [] simulatedCapacities = null;
    
    public Builder(Configuration conf) {
      this.conf = conf;
    }
    
    /**
     * Default: 0
     */
    public Builder nameNodePort(int val) {
      this.nameNodePort = val;
      return this;
    }

    /**
     * Default: 1
     */
    public Builder numDataNodes(int val) {
      this.numDataNodes = val;
      return this;
    }

    /**
     * Default: true
     */
    public Builder format(boolean val) {
      this.format = val;
      return this;
    }

    /**
     * Default: true
     */
    public Builder manageNameDfsDirs(boolean val) {
      this.manageNameDfsDirs = val;
      return this;
    }

    /**
     * Default: true
     */
    public Builder manageDataDfsDirs(boolean val) {
      this.manageDataDfsDirs = val;
      return this;
    }

    /**
     * Default: null
     */
    public Builder startupOption(StartupOption val) {
      this.option = val;
      return this;
    }

    /**
     * Default: null
     */
    public Builder racks(String[] val) {
      this.racks = val;
      return this;
    }

    /**
     * Default: null
     */
    public Builder hosts(String[] val) {
      this.hosts = val;
      return this;
    }

    /**
     * Default: null
     */
    public Builder simulatedCapacities(long[] val) {
      this.simulatedCapacities = val;
      return this;
    }
    
    /**
     * Construct the actual MiniDFSCluster
     */
    public MiniDFSCluster build() throws IOException {
      return new MiniDFSCluster(this);
    }
  }
  
  /**
   * Used by builder to create and return an instance of MiniDFSCluster
   */
  private MiniDFSCluster(Builder builder) throws IOException {
    initMiniDFSCluster(builder.nameNodePort,
                       builder.conf,
                       builder.numDataNodes,
                       builder.format,
                       builder.manageNameDfsDirs,
                       builder.manageDataDfsDirs,
                       builder.option,
                       builder.racks,
                       builder.hosts,
                       builder.simulatedCapacities);
  }
  
  public class DataNodeProperties {
    DataNode datanode;
    Configuration conf;
    String[] dnArgs;

    DataNodeProperties(DataNode node, Configuration conf, String[] args) {
      this.datanode = node;
      this.conf = conf;
      this.dnArgs = args;
    }
  }
  private static final Log LOG = LogFactory.getLog(MiniDFSCluster.class);

  private URI myUri = null;
  private Configuration conf;
  private NameNode nameNode;
  private int numDataNodes;
  private ArrayList<DataNodeProperties> dataNodes = 
                         new ArrayList<DataNodeProperties>();
  private File base_dir;
  private File data_dir;
  
  public final static String FINALIZED_DIR_NAME = "/current/finalized/";
  
  
  /**
   * This null constructor is used only when wishing to start a data node cluster
   * without a name node (ie when the name node is started elsewhere).
   */
  public MiniDFSCluster() {
  }
  
  /**
   * Modify the config and start up the servers with the given operation.
   * Servers will be started on free ports.
   * <p>
   * The caller must manage the creation of NameNode and DataNode directories
   * and have already set dfs.name.dir and dfs.data.dir in the given conf.
   * 
   * @param conf the base configuration to use in starting the servers.  This
   *          will be modified as necessary.
   * @param numDataNodes Number of DataNodes to start; may be zero
   * @param nameNodeOperation the operation with which to start the servers.  If null
   *          or StartupOption.FORMAT, then StartupOption.REGULAR will be used.
   */
  @Deprecated // in 22 to be removed in 24. Use MiniDFSCluster.Builder instead
  public MiniDFSCluster(Configuration conf,
                        int numDataNodes,
                        StartupOption nameNodeOperation) throws IOException {
    this(0, conf, numDataNodes, false, false, false,  nameNodeOperation, 
          null, null, null);
  }
  
  /**
   * Modify the config and start up the servers.  The rpc and info ports for
   * servers are guaranteed to use free ports.
   * <p>
   * NameNode and DataNode directory creation and configuration will be
   * managed by this class.
   *
   * @param conf the base configuration to use in starting the servers.  This
   *          will be modified as necessary.
   * @param numDataNodes Number of DataNodes to start; may be zero
   * @param format if true, format the NameNode and DataNodes before starting up
   * @param racks array of strings indicating the rack that each DataNode is on
   */
  @Deprecated // in 22 to be removed in 24. Use MiniDFSCluster.Builder instead
  public MiniDFSCluster(Configuration conf,
                        int numDataNodes,
                        boolean format,
                        String[] racks) throws IOException {
    this(0, conf, numDataNodes, format, true, true,  null, racks, null, null);
  }
  
  /**
   * Modify the config and start up the servers.  The rpc and info ports for
   * servers are guaranteed to use free ports.
   * <p>
   * NameNode and DataNode directory creation and configuration will be
   * managed by this class.
   *
   * @param conf the base configuration to use in starting the servers.  This
   *          will be modified as necessary.
   * @param numDataNodes Number of DataNodes to start; may be zero
   * @param format if true, format the NameNode and DataNodes before starting up
   * @param racks array of strings indicating the rack that each DataNode is on
   * @param hosts array of strings indicating the hostname for each DataNode
   */
  @Deprecated // in 22 to be removed in 24. Use MiniDFSCluster.Builder instead
  public MiniDFSCluster(Configuration conf,
                        int numDataNodes,
                        boolean format,
                        String[] racks, String[] hosts) throws IOException {
    this(0, conf, numDataNodes, format, true, true, null, racks, hosts, null);
  }
  
  /**
   * NOTE: if possible, the other constructors that don't have nameNode port 
   * parameter should be used as they will ensure that the servers use free ports.
   * <p>
   * Modify the config and start up the servers.  
   * 
   * @param nameNodePort suggestion for which rpc port to use.  caller should
   *          use getNameNodePort() to get the actual port used.
   * @param conf the base configuration to use in starting the servers.  This
   *          will be modified as necessary.
   * @param numDataNodes Number of DataNodes to start; may be zero
   * @param format if true, format the NameNode and DataNodes before starting up
   * @param manageDfsDirs if true, the data directories for servers will be
   *          created and dfs.name.dir and dfs.data.dir will be set in the conf
   * @param operation the operation with which to start the servers.  If null
   *          or StartupOption.FORMAT, then StartupOption.REGULAR will be used.
   * @param racks array of strings indicating the rack that each DataNode is on
   */
  @Deprecated // in 22 to be removed in 24. Use MiniDFSCluster.Builder instead
  public MiniDFSCluster(int nameNodePort, 
                        Configuration conf,
                        int numDataNodes,
                        boolean format,
                        boolean manageDfsDirs,
                        StartupOption operation,
                        String[] racks) throws IOException {
    this(nameNodePort, conf, numDataNodes, format, manageDfsDirs, manageDfsDirs,
         operation, racks, null, null);
  }

  /**
   * NOTE: if possible, the other constructors that don't have nameNode port 
   * parameter should be used as they will ensure that the servers use free ports.
   * <p>
   * Modify the config and start up the servers.  
   * 
   * @param nameNodePort suggestion for which rpc port to use.  caller should
   *          use getNameNodePort() to get the actual port used.
   * @param conf the base configuration to use in starting the servers.  This
   *          will be modified as necessary.
   * @param numDataNodes Number of DataNodes to start; may be zero
   * @param format if true, format the NameNode and DataNodes before starting up
   * @param manageDfsDirs if true, the data directories for servers will be
   *          created and dfs.name.dir and dfs.data.dir will be set in the conf
   * @param operation the operation with which to start the servers.  If null
   *          or StartupOption.FORMAT, then StartupOption.REGULAR will be used.
   * @param racks array of strings indicating the rack that each DataNode is on
   * @param simulatedCapacities array of capacities of the simulated data nodes
   */
  @Deprecated // in 22 to be removed in 24. Use MiniDFSCluster.Builder instead
  public MiniDFSCluster(int nameNodePort, 
                        Configuration conf,
                        int numDataNodes,
                        boolean format,
                        boolean manageDfsDirs,
                        StartupOption operation,
                        String[] racks,
                        long[] simulatedCapacities) throws IOException {
    this(nameNodePort, conf, numDataNodes, format, manageDfsDirs, manageDfsDirs,
          operation, racks, null, simulatedCapacities);
  }
  
  /**
   * NOTE: if possible, the other constructors that don't have nameNode port 
   * parameter should be used as they will ensure that the servers use free ports.
   * <p>
   * Modify the config and start up the servers.  
   * 
   * @param nameNodePort suggestion for which rpc port to use.  caller should
   *          use getNameNodePort() to get the actual port used.
   * @param conf the base configuration to use in starting the servers.  This
   *          will be modified as necessary.
   * @param numDataNodes Number of DataNodes to start; may be zero
   * @param format if true, format the NameNode and DataNodes before starting up
   * @param manageNameDfsDirs if true, the data directories for servers will be
   *          created and dfs.name.dir and dfs.data.dir will be set in the conf
   * @param manageDataDfsDirs if true, the data directories for datanodes will
   *          be created and dfs.data.dir set to same in the conf
   * @param operation the operation with which to start the servers.  If null
   *          or StartupOption.FORMAT, then StartupOption.REGULAR will be used.
   * @param racks array of strings indicating the rack that each DataNode is on
   * @param hosts array of strings indicating the hostnames of each DataNode
   * @param simulatedCapacities array of capacities of the simulated data nodes
   */
  @Deprecated // in 22 to be removed in 24. Use MiniDFSCluster.Builder instead
  public MiniDFSCluster(int nameNodePort, 
                        Configuration conf,
                        int numDataNodes,
                        boolean format,
                        boolean manageNameDfsDirs,
                        boolean manageDataDfsDirs,
                        StartupOption operation,
                        String[] racks, String hosts[],
                        long[] simulatedCapacities) throws IOException {
    initMiniDFSCluster(nameNodePort, conf, numDataNodes, format,
        manageNameDfsDirs, manageDataDfsDirs, operation, racks, hosts,
        simulatedCapacities);
  }

  private void initMiniDFSCluster(int nameNodePort, Configuration conf,
      int numDataNodes, boolean format, boolean manageNameDfsDirs,
      boolean manageDataDfsDirs, StartupOption operation, String[] racks,
      String[] hosts, long[] simulatedCapacities) throws IOException {
    this.conf = conf;
    base_dir = new File(getBaseDirectory());
    data_dir = new File(base_dir, "data");
    
    // use alternate RPC engine if spec'd
    String rpcEngineName = System.getProperty("hdfs.rpc.engine");
    if (rpcEngineName != null && !"".equals(rpcEngineName)) {
      
      System.out.println("HDFS using RPCEngine: "+rpcEngineName);
      try {
        Class<?> rpcEngine = conf.getClassByName(rpcEngineName);
        setRpcEngine(conf, NamenodeProtocols.class, rpcEngine);
        setRpcEngine(conf, NamenodeProtocol.class, rpcEngine);
        setRpcEngine(conf, ClientProtocol.class, rpcEngine);
        setRpcEngine(conf, DatanodeProtocol.class, rpcEngine);
        setRpcEngine(conf, RefreshAuthorizationPolicyProtocol.class, rpcEngine);
        setRpcEngine(conf, RefreshUserMappingsProtocol.class, rpcEngine);
      } catch (ClassNotFoundException e) {
        throw new RuntimeException(e);
      }

      // disable service authorization, as it does not work with tunnelled RPC
      conf.setBoolean(CommonConfigurationKeys.HADOOP_SECURITY_AUTHORIZATION,
                      false);
    }

    // Setup the NameNode configuration
    FileSystem.setDefaultUri(conf, "hdfs://localhost:"+ Integer.toString(nameNodePort));
    conf.set(DFSConfigKeys.DFS_NAMENODE_HTTP_ADDRESS_KEY, "127.0.0.1:0");  
    if (manageNameDfsDirs) {
      conf.set(DFSConfigKeys.DFS_NAMENODE_NAME_DIR_KEY,
          fileAsURI(new File(base_dir, "name1"))+","+
          fileAsURI(new File(base_dir, "name2")));
      conf.set(DFSConfigKeys.DFS_NAMENODE_CHECKPOINT_DIR_KEY,
          fileAsURI(new File(base_dir, "namesecondary1"))+","+
          fileAsURI(new File(base_dir, "namesecondary2")));
    }
    
    int replication = conf.getInt("dfs.replication", 3);
    conf.setInt("dfs.replication", Math.min(replication, numDataNodes));
    conf.setInt(DFSConfigKeys.DFS_NAMENODE_SAFEMODE_EXTENSION_KEY, 0);
    conf.setInt("dfs.namenode.decommission.interval", 3); // 3 second
    
    // Format and clean out DataNode directories
    if (format) {
      if (data_dir.exists() && !FileUtil.fullyDelete(data_dir)) {
        throw new IOException("Cannot remove data directory: " + data_dir);
      }
      NameNode.format(conf); 
    }
    
    // Start the NameNode
    String[] args = (operation == null ||
                     operation == StartupOption.FORMAT ||
                     operation == StartupOption.REGULAR) ?
      new String[] {} : new String[] {operation.getName()};
    conf.setClass(DFSConfigKeys.NET_TOPOLOGY_NODE_SWITCH_MAPPING_IMPL_KEY, 
                   StaticMapping.class, DNSToSwitchMapping.class);
    nameNode = NameNode.createNameNode(args, conf);
    
    // Start the DataNodes
    startDataNodes(conf, numDataNodes, manageDataDfsDirs, 
                    operation, racks, hosts, simulatedCapacities);
    waitClusterUp();

    //make sure ProxyUsers uses the latest conf
    ProxyUsers.refreshSuperUserGroupsConfiguration(conf);
  }
  
  private void setRpcEngine(Configuration conf, Class<?> protocol, Class<?> engine) {
    conf.setClass("rpc.engine."+protocol.getName(), engine, Object.class);
  }

  /**
   * 
   * @return URI of this MiniDFSCluster
   */
  public URI getURI() {
    return myUri;
  }

  /**
   * Get configuration.
   * @return Configuration of this MiniDFSCluster
   */
  public Configuration getConfiguration() {
    return conf;
  }

  /**
   * wait for the cluster to get out of 
   * safemode.
   */
  public void waitClusterUp() {
    if (numDataNodes > 0) {
      while (!isClusterUp()) {
        try {
          LOG.warn("Waiting for the Mini HDFS Cluster to start...");
          Thread.sleep(1000);
        } catch (InterruptedException e) {
        }
      }
    }
  }

  /**
   * Modify the config and start up additional DataNodes.  The info port for
   * DataNodes is guaranteed to use a free port.
   *  
   *  Data nodes can run with the name node in the mini cluster or
   *  a real name node. For example, running with a real name node is useful
   *  when running simulated data nodes with a real name node.
   *  If minicluster's name node is null assume that the conf has been
   *  set with the right address:port of the name node.
   *
   * @param conf the base configuration to use in starting the DataNodes.  This
   *          will be modified as necessary.
   * @param numDataNodes Number of DataNodes to start; may be zero
   * @param manageDfsDirs if true, the data directories for DataNodes will be
   *          created and dfs.data.dir will be set in the conf
   * @param operation the operation with which to start the DataNodes.  If null
   *          or StartupOption.FORMAT, then StartupOption.REGULAR will be used.
   * @param racks array of strings indicating the rack that each DataNode is on
   * @param hosts array of strings indicating the hostnames for each DataNode
   * @param simulatedCapacities array of capacities of the simulated data nodes
   *
   * @throws IllegalStateException if NameNode has been shutdown
   */
  public synchronized void startDataNodes(Configuration conf, int numDataNodes, 
                             boolean manageDfsDirs, StartupOption operation, 
                             String[] racks, String[] hosts,
                             long[] simulatedCapacities) throws IOException {

    int curDatanodesNum = dataNodes.size();
    // for mincluster's the default initialDelay for BRs is 0
    if (conf.get("dfs.blockreport.initialDelay") == null) {
      conf.setLong("dfs.blockreport.initialDelay", 0);
    }
    // If minicluster's name node is null assume that the conf has been
    // set with the right address:port of the name node.
    //
    if (nameNode != null) { // set conf from the name node
      InetSocketAddress nnAddr = nameNode.getNameNodeAddress(); 
      int nameNodePort = nnAddr.getPort(); 
      try {
	  myUri = new URI("hdfs://"+ nnAddr.getHostName() + ":" +
			  Integer.toString(nameNodePort));
      } catch (URISyntaxException e) {
	  throw new IOException("Couldn't parse own URI", e);
      }
      FileSystem.setDefaultUri(conf, myUri);
    }

    if (racks != null && numDataNodes > racks.length ) {
      throw new IllegalArgumentException( "The length of racks [" + racks.length
          + "] is less than the number of datanodes [" + numDataNodes + "].");
    }
    if (hosts != null && numDataNodes > hosts.length ) {
      throw new IllegalArgumentException( "The length of hosts [" + hosts.length
          + "] is less than the number of datanodes [" + numDataNodes + "].");
    }
    //Generate some hostnames if required
    if (racks != null && hosts == null) {
      System.out.println("Generating host names for datanodes");
      hosts = new String[numDataNodes];
      for (int i = curDatanodesNum; i < curDatanodesNum + numDataNodes; i++) {
        hosts[i - curDatanodesNum] = "host" + i + ".foo.com";
      }
    }

    if (simulatedCapacities != null 
        && numDataNodes > simulatedCapacities.length) {
      throw new IllegalArgumentException( "The length of simulatedCapacities [" 
          + simulatedCapacities.length
          + "] is less than the number of datanodes [" + numDataNodes + "].");
    }

    // Set up the right ports for the datanodes
    conf.set("dfs.datanode.address", "127.0.0.1:0");
    conf.set("dfs.datanode.http.address", "127.0.0.1:0");
    conf.set("dfs.datanode.ipc.address", "127.0.0.1:0");
    

    String [] dnArgs = (operation == null ||
                        operation != StartupOption.ROLLBACK) ?
        null : new String[] {operation.getName()};
    
    
    for (int i = curDatanodesNum; i < curDatanodesNum+numDataNodes; i++) {
      Configuration dnConf = new HdfsConfiguration(conf);
      if (manageDfsDirs) {
        File dir1 = new File(data_dir, "data"+(2*i+1));
        File dir2 = new File(data_dir, "data"+(2*i+2));
        dir1.mkdirs();
        dir2.mkdirs();
        if (!dir1.isDirectory() || !dir2.isDirectory()) { 
          throw new IOException("Mkdirs failed to create directory for DataNode "
                                + i + ": " + dir1 + " or " + dir2);
        }
        String dirs = fileAsURI(dir1) + "," + fileAsURI(dir2);
        dnConf.set(DFSConfigKeys.DFS_DATANODE_DATA_DIR_KEY, dirs);
        conf.set(DFSConfigKeys.DFS_DATANODE_DATA_DIR_KEY, dirs);
      }
      if (simulatedCapacities != null) {
        dnConf.setBoolean("dfs.datanode.simulateddatastorage", true);
        dnConf.setLong(SimulatedFSDataset.CONFIG_PROPERTY_CAPACITY,
            simulatedCapacities[i-curDatanodesNum]);
      }
      System.out.println("Starting DataNode " + i + " with "
                         + DFSConfigKeys.DFS_DATANODE_DATA_DIR_KEY + ": "
                         + dnConf.get(DFSConfigKeys.DFS_DATANODE_DATA_DIR_KEY));
      if (hosts != null) {
        dnConf.set(DFSConfigKeys.DFS_DATANODE_HOST_NAME_KEY, hosts[i - curDatanodesNum]);
        System.out.println("Starting DataNode " + i + " with hostname set to: " 
                           + dnConf.get(DFSConfigKeys.DFS_DATANODE_HOST_NAME_KEY));
      }
      if (racks != null) {
        String name = hosts[i - curDatanodesNum];
        System.out.println("Adding node with hostname : " + name + " to rack "+
                            racks[i-curDatanodesNum]);
        StaticMapping.addNodeToRack(name,
                                    racks[i-curDatanodesNum]);
      }
      Configuration newconf = new HdfsConfiguration(dnConf); // save config
      if (hosts != null) {
        NetUtils.addStaticResolution(hosts[i - curDatanodesNum], "localhost");
      }
      DataNode dn = DataNode.instantiateDataNode(dnArgs, dnConf);
      if(dn == null)
        throw new IOException("Cannot start DataNode in "
            + dnConf.get(DFSConfigKeys.DFS_DATANODE_DATA_DIR_KEY));
      //since the HDFS does things based on IP:port, we need to add the mapping
      //for IP:port to rackId
      String ipAddr = dn.getSelfAddr().getAddress().getHostAddress();
      if (racks != null) {
        int port = dn.getSelfAddr().getPort();
        System.out.println("Adding node with IP:port : " + ipAddr + ":" + port+
                            " to rack " + racks[i-curDatanodesNum]);
        StaticMapping.addNodeToRack(ipAddr + ":" + port,
                                  racks[i-curDatanodesNum]);
      }
      DataNode.runDatanodeDaemon(dn);
      dataNodes.add(new DataNodeProperties(dn, newconf, dnArgs));
    }
    curDatanodesNum += numDataNodes;
    this.numDataNodes += numDataNodes;
    waitActive();
  }
  
  
  
  /**
   * Modify the config and start up the DataNodes.  The info port for
   * DataNodes is guaranteed to use a free port.
   *
   * @param conf the base configuration to use in starting the DataNodes.  This
   *          will be modified as necessary.
   * @param numDataNodes Number of DataNodes to start; may be zero
   * @param manageDfsDirs if true, the data directories for DataNodes will be
   *          created and dfs.data.dir will be set in the conf
   * @param operation the operation with which to start the DataNodes.  If null
   *          or StartupOption.FORMAT, then StartupOption.REGULAR will be used.
   * @param racks array of strings indicating the rack that each DataNode is on
   *
   * @throws IllegalStateException if NameNode has been shutdown
   */
  
  public void startDataNodes(Configuration conf, int numDataNodes, 
      boolean manageDfsDirs, StartupOption operation, 
      String[] racks
      ) throws IOException {
    startDataNodes( conf,  numDataNodes, manageDfsDirs,  operation, racks, null, null);
  }
  
  /**
   * Modify the config and start up additional DataNodes.  The info port for
   * DataNodes is guaranteed to use a free port.
   *  
   *  Data nodes can run with the name node in the mini cluster or
   *  a real name node. For example, running with a real name node is useful
   *  when running simulated data nodes with a real name node.
   *  If minicluster's name node is null assume that the conf has been
   *  set with the right address:port of the name node.
   *
   * @param conf the base configuration to use in starting the DataNodes.  This
   *          will be modified as necessary.
   * @param numDataNodes Number of DataNodes to start; may be zero
   * @param manageDfsDirs if true, the data directories for DataNodes will be
   *          created and dfs.data.dir will be set in the conf
   * @param operation the operation with which to start the DataNodes.  If null
   *          or StartupOption.FORMAT, then StartupOption.REGULAR will be used.
   * @param racks array of strings indicating the rack that each DataNode is on
   * @param simulatedCapacities array of capacities of the simulated data nodes
   *
   * @throws IllegalStateException if NameNode has been shutdown
   */
  public void startDataNodes(Configuration conf, int numDataNodes, 
                             boolean manageDfsDirs, StartupOption operation, 
                             String[] racks,
                             long[] simulatedCapacities) throws IOException {
    startDataNodes(conf, numDataNodes, manageDfsDirs, operation, racks, null,
                   simulatedCapacities);
    
  }
  /**
   * If the NameNode is running, attempt to finalize a previous upgrade.
   * When this method return, the NameNode should be finalized, but
   * DataNodes may not be since that occurs asynchronously.
   *
   * @throws IllegalStateException if the Namenode is not running.
   */
  public void finalizeCluster(Configuration conf) throws Exception {
    if (nameNode == null) {
      throw new IllegalStateException("Attempting to finalize "
                                      + "Namenode but it is not running");
    }
    ToolRunner.run(new DFSAdmin(conf), new String[] {"-finalizeUpgrade"});
  }
  
  /**
   * Gets the started NameNode.  May be null.
   */
  public NameNode getNameNode() {
    return nameNode;
  }
  
  /**
   * Return the {@link FSNamesystem} object.
   * @return {@link FSNamesystem} object.
   */
  public FSNamesystem getNamesystem() {
    return NameNodeAdapter.getNamesystem(nameNode);
  }

  /**
   * Gets a list of the started DataNodes.  May be empty.
   */
  public ArrayList<DataNode> getDataNodes() {
    ArrayList<DataNode> list = new ArrayList<DataNode>();
    for (int i = 0; i < dataNodes.size(); i++) {
      DataNode node = dataNodes.get(i).datanode;
      list.add(node);
    }
    return list;
  }
  
  /** @return the datanode having the ipc server listen port */
  public DataNode getDataNode(int ipcPort) {
    for(DataNode dn : getDataNodes()) {
      if (dn.ipcServer.getListenerAddress().getPort() == ipcPort) {
        return dn;
      }
    }
    return null;
  }

  /**
   * Gets the rpc port used by the NameNode, because the caller 
   * supplied port is not necessarily the actual port used.
   */     
  public int getNameNodePort() {
    return nameNode.getNameNodeAddress().getPort();
  }
    
  /**
   * Shut down the servers that are up.
   */
  public void shutdown() {
    System.out.println("Shutting down the Mini HDFS Cluster");
    shutdownDataNodes();
    if (nameNode != null) {
      nameNode.stop();
      nameNode.join();
      nameNode = null;
    }
  }
  
  /**
   * Shutdown all DataNodes started by this class.  The NameNode
   * is left running so that new DataNodes may be started.
   */
  public void shutdownDataNodes() {
    for (int i = dataNodes.size()-1; i >= 0; i--) {
      LOG.info("Shutting down DataNode " + i);
      DataNode dn = dataNodes.remove(i).datanode;
      dn.shutdown();
      numDataNodes--;
    }
  }

  /**
   * Shutdown namenode.
   */
  public synchronized void shutdownNameNode() {
    if (nameNode != null) {
      System.out.println("Shutting down the namenode");
      nameNode.stop();
      nameNode.join();
      nameNode = null;
    }
  }

  /**
   * Restart namenode.
   */
  public synchronized void restartNameNode() throws IOException {
    shutdownNameNode();
    nameNode = NameNode.createNameNode(new String[] {}, conf);
    waitClusterUp();
    System.out.println("Restarted the namenode");
    int failedCount = 0;
    while (true) {
      try {
        waitActive();
        break;
      } catch (IOException e) {
        failedCount++;
        // Cached RPC connection to namenode, if any, is expected to fail once
        if (failedCount > 1) {
          System.out.println("Tried waitActive() " + failedCount
              + " time(s) and failed, giving up.  "
              + StringUtils.stringifyException(e));
          throw e;
        }
      }
    }
    System.out.println("Cluster is active");
  }

  /**
   * Return the contents of the given block on the given datanode.
   *
   * @param The index of the datanode
   * @param The name of the block
   * @throws IOException on error accessing the file for the given block
   * @return The contents of the block file, null if none found
   */
  public String readBlockOnDataNode(int i, String blockName) throws IOException {
    assert (i >= 0 && i < dataNodes.size()) : "Invalid datanode "+i;

    // Each datanode has multiple data dirs, check each
    for (int dn = i*2; dn < i*2+2; dn++) {
      File dataDir = new File(getBaseDirectory() + "data");
      File blockFile = new File(dataDir,
          "data" + (dn+1) + FINALIZED_DIR_NAME + blockName);
      if (blockFile.exists()) {
        return DFSTestUtil.readFile(blockFile);
      }
    }
    return null;
  }

  /**
   * Corrupt a block on all datanodes.
   *
   * @param The name of the block
   * @throws IOException on error accessing the given block.
   * @return The number of block files corrupted.
   */  
  public int corruptBlockOnDataNodes(String blockName) throws IOException {
    int blocksCorrupted = 0;
    for (int i=0; i < dataNodes.size(); i++) {
      if (corruptReplica(blockName, i)) {
        blocksCorrupted++;
      }
    }
    return blocksCorrupted;
  }

  /**
   * Corrupt a block on a particular datanode.
   *
   * @param The index of the datanode
   * @param The name of the block
   * @throws IOException on error accessing the given block or if
   * the contents of the block (on the same datanode) differ.
   * @return true if a replica was corrupted, false otherwise
   */
  public boolean corruptReplica(String blockName, int i) throws IOException {
    Random random = new Random();
    assert (i >= 0 && i < dataNodes.size()) : "Invalid datanode "+i;
    int filesCorrupted = 0;

    // Each datanode has multiple data dirs, check each
    for (int dn = i*2; dn < i*2+2; dn++) {
      File dataDir = new File(getBaseDirectory() + "data");
      File blockFile = new File(dataDir,
          "data" + (dn+1) + FINALIZED_DIR_NAME + blockName);

      // Corrupt the replica by writing some bytes into a random offset
      if (blockFile.exists()) { 
        System.out.println("Corrupting " + blockFile);
        RandomAccessFile raFile = new RandomAccessFile(blockFile, "rw");
        FileChannel channel = raFile.getChannel();
        String badString = "BADBAD";
        int rand = random.nextInt((int)channel.size()/2);
        raFile.seek(rand);
        raFile.write(badString.getBytes());
        raFile.close();
        filesCorrupted++;
      }
    }
    assert filesCorrupted == 0 || filesCorrupted == 1
      : "Unexpected # block files";
    return filesCorrupted == 1;
  }

  /*
   * Shutdown a particular datanode
   */
  public synchronized DataNodeProperties stopDataNode(int i) {
    if (i < 0 || i >= dataNodes.size()) {
      return null;
    }
    DataNodeProperties dnprop = dataNodes.remove(i);
    DataNode dn = dnprop.datanode;
    System.out.println("MiniDFSCluster Stopping DataNode " + 
                       dn.dnRegistration.getName() +
                       " from a total of " + (dataNodes.size() + 1) + 
                       " datanodes.");
    dn.shutdown();
    numDataNodes--;
    return dnprop;
  }

  /*
   * Shutdown a datanode by name.
   */
  public synchronized DataNodeProperties stopDataNode(String name) {
    int i;
    for (i = 0; i < dataNodes.size(); i++) {
      DataNode dn = dataNodes.get(i).datanode;
      if (dn.dnRegistration.getName().equals(name)) {
        break;
      }
    }
    return stopDataNode(i);
  }

  /**
   * Restart a datanode
   * @param dnprop datanode's property
   * @return true if restarting is successful
   * @throws IOException
   */
  public boolean restartDataNode(DataNodeProperties dnprop) throws IOException {
    return restartDataNode(dnprop, false);
  }

  /**
   * Restart a datanode, on the same port if requested
   * @param dnprop the datanode to restart
   * @param keepPort whether to use the same port 
   * @return true if restarting is successful
   * @throws IOException
   */
  public synchronized boolean restartDataNode(DataNodeProperties dnprop,
      boolean keepPort) throws IOException {
    Configuration conf = dnprop.conf;
    String[] args = dnprop.dnArgs;
    Configuration newconf = new HdfsConfiguration(conf); // save cloned config
    if (keepPort) {
      InetSocketAddress addr = dnprop.datanode.getSelfAddr();
      conf.set("dfs.datanode.address", addr.getAddress().getHostAddress() + ":"
          + addr.getPort());
    }
    dataNodes.add(new DataNodeProperties(DataNode.createDataNode(args, conf),
        newconf, args));
    numDataNodes++;
    return true;
  }

  /*
   * Restart a particular datanode, use newly assigned port
   */
  public boolean restartDataNode(int i) throws IOException {
    return restartDataNode(i, false);
  }

  /*
   * Restart a particular datanode, on the same port if keepPort is true
   */
  public synchronized boolean restartDataNode(int i, boolean keepPort)
      throws IOException {
    DataNodeProperties dnprop = stopDataNode(i);
    if (dnprop == null) {
      return false;
    } else {
      return restartDataNode(dnprop, keepPort);
    }
  }

  /*
   * Restart all datanodes, on the same ports if keepPort is true
   */
  public synchronized boolean restartDataNodes(boolean keepPort)
      throws IOException {
    for (int i = dataNodes.size() - 1; i >= 0; i--) {
      if (!restartDataNode(i, keepPort))
        return false;
      System.out.println("Restarted DataNode " + i);
    }
    return true;
  }

  /*
   * Restart all datanodes, use newly assigned ports
   */
  public boolean restartDataNodes() throws IOException {
    return restartDataNodes(false);
  }
  
  /**
   * Returns true if the NameNode is running and is out of Safe Mode.
   */
  public boolean isClusterUp() {
    if (nameNode == null) {
      return false;
    }
    long[] sizes = nameNode.getStats();
    boolean isUp = false;
    synchronized (this) {
      isUp = (!nameNode.isInSafeMode() && sizes[0] != 0);
    }
    return isUp;
  }
  
  /**
   * Returns true if there is at least one DataNode running.
   */
  public boolean isDataNodeUp() {
    if (dataNodes == null || dataNodes.size() == 0) {
      return false;
    }
    return true;
  }
  
  /**
   * Get a client handle to the DFS cluster.
   */
  public FileSystem getFileSystem() throws IOException {
    return FileSystem.get(conf);
  }
  

  /**
   * Get another FileSystem instance that is different from FileSystem.get(conf).
   * This simulating different threads working on different FileSystem instances.
   */
  public FileSystem getNewFileSystemInstance() throws IOException {
    return FileSystem.newInstance(conf);
  }
  
  /**
   * @return a {@link HftpFileSystem} object.
   */
  public HftpFileSystem getHftpFileSystem() throws IOException {
    final String str = "hftp://"
        + conf.get(DFSConfigKeys.DFS_NAMENODE_HTTP_ADDRESS_KEY);
    try {
      return (HftpFileSystem)FileSystem.get(new URI(str), conf); 
    } catch (URISyntaxException e) {
      throw new IOException(e);
    }
  }

  /**
   *  @return a {@link HftpFileSystem} object as specified user. 
   */
  public HftpFileSystem getHftpFileSystemAs(final String username,
      final Configuration conf, final String... groups
      ) throws IOException, InterruptedException {
    final UserGroupInformation ugi = UserGroupInformation.createUserForTesting(
        username, groups);
    return ugi.doAs(new PrivilegedExceptionAction<HftpFileSystem>() {
      @Override
      public HftpFileSystem run() throws Exception {
        return getHftpFileSystem();
      }
    });
  }

  /**
   * Get the directories where the namenode stores its image.
   */
  public Collection<URI> getNameDirs() {
    return FSNamesystem.getNamespaceDirs(conf);
  }

  /**
   * Get the directories where the namenode stores its edits.
   */
  public Collection<URI> getNameEditsDirs() {
    return FSNamesystem.getNamespaceEditsDirs(conf);
  }

  /**
   * Wait until the cluster is active and running.
   */
  public void waitActive() throws IOException {
    if (nameNode == null) {
      return;
    }
    InetSocketAddress addr = new InetSocketAddress("localhost",
                                                   getNameNodePort());
    DFSClient client = new DFSClient(addr, conf);

    // make sure all datanodes have registered and sent heartbeat
    while (shouldWait(client.datanodeReport(DatanodeReportType.LIVE))) {
      try {
        Thread.sleep(100);
      } catch (InterruptedException e) {
      }
    }

    client.close();
  }
  
  private synchronized boolean shouldWait(DatanodeInfo[] dnInfo) {
    if (dnInfo.length != numDataNodes) {
      return true;
    }
    // make sure all datanodes have sent first heartbeat to namenode,
    // using (capacity == 0) as proxy.
    for (DatanodeInfo dn : dnInfo) {
      if (dn.getCapacity() == 0) {
        return true;
      }
    }
    return false;
  }

  public void formatDataNodeDirs() throws IOException {
    base_dir = new File(getBaseDirectory());
    data_dir = new File(base_dir, "data");
    if (data_dir.exists() && !FileUtil.fullyDelete(data_dir)) {
      throw new IOException("Cannot remove data directory: " + data_dir);
    }
  }
  
  /**
   * 
   * @param dataNodeIndex - data node whose block report is desired - the index is same as for getDataNodes()
   * @return the block report for the specified data node
   */
  public Iterable<Block> getBlockReport(int dataNodeIndex) {
    if (dataNodeIndex < 0 || dataNodeIndex > dataNodes.size()) {
      throw new IndexOutOfBoundsException();
    }
    return dataNodes.get(dataNodeIndex).datanode.getFSDataset().getBlockReport();
  }
  
  
  /**
   * 
   * @return block reports from all data nodes
   *    BlockListAsLongs is indexed in the same order as the list of datanodes returned by getDataNodes()
   */
  public Iterable<Block>[] getAllBlockReports() {
    int numDataNodes = dataNodes.size();
    Iterable<Block>[] result = new BlockListAsLongs[numDataNodes];
    for (int i = 0; i < numDataNodes; ++i) {
     result[i] = getBlockReport(i);
    }
    return result;
  }
  
  
  /**
   * This method is valid only if the data nodes have simulated data
   * @param dataNodeIndex - data node i which to inject - the index is same as for getDataNodes()
   * @param blocksToInject - the blocks
   * @throws IOException
   *              if not simulatedFSDataset
   *             if any of blocks already exist in the data node
   *   
   */
  public void injectBlocks(int dataNodeIndex, Iterable<Block> blocksToInject) throws IOException {
    if (dataNodeIndex < 0 || dataNodeIndex > dataNodes.size()) {
      throw new IndexOutOfBoundsException();
    }
    FSDatasetInterface dataSet = dataNodes.get(dataNodeIndex).datanode.getFSDataset();
    if (!(dataSet instanceof SimulatedFSDataset)) {
      throw new IOException("injectBlocks is valid only for SimilatedFSDataset");
    }
    SimulatedFSDataset sdataset = (SimulatedFSDataset) dataSet;
    sdataset.injectBlocks(blocksToInject);
    dataNodes.get(dataNodeIndex).datanode.scheduleBlockReport(0);
  }
  
  /**
   * This method is valid only if the data nodes have simulated data
   * @param blocksToInject - blocksToInject[] is indexed in the same order as the list 
   *             of datanodes returned by getDataNodes()
   * @throws IOException
   *             if not simulatedFSDataset
   *             if any of blocks already exist in the data nodes
   *             Note the rest of the blocks are not injected.
   */
  public void injectBlocks(Iterable<Block>[] blocksToInject) throws IOException {
    if (blocksToInject.length >  dataNodes.size()) {
      throw new IndexOutOfBoundsException();
    }
    for (int i = 0; i < blocksToInject.length; ++i) {
     injectBlocks(i, blocksToInject[i]);
    }
  }

  /**
   * Set the softLimit and hardLimit of client lease periods
   */
  void setLeasePeriod(long soft, long hard) {
    final FSNamesystem namesystem = getNamesystem();
    namesystem.leaseManager.setLeasePeriod(soft, hard);
    namesystem.lmthread.interrupt();
  }

  /**
   * Returns the current set of datanodes
   */
  DataNode[] listDataNodes() {
    DataNode[] list = new DataNode[dataNodes.size()];
    for (int i = 0; i < dataNodes.size(); i++) {
      list[i] = dataNodes.get(i).datanode;
    }
    return list;
  }

  /**
   * Access to the data directory used for Datanodes
   */
  public String getDataDirectory() {
    return data_dir.getAbsolutePath();
  }

  public static String getBaseDirectory() {
    return System.getProperty("test.build.data", "build/test/data") + "/dfs/";
  }
}

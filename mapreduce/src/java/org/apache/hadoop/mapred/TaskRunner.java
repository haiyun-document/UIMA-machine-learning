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
package org.apache.hadoop.mapred;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.security.PrivilegedExceptionAction;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapreduce.MRConfig;
import org.apache.hadoop.mapreduce.MRJobConfig;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.filecache.TaskDistributedCacheManager;
import org.apache.hadoop.mapreduce.filecache.TrackerDistributedCacheManager;
import org.apache.hadoop.mapreduce.security.TokenCache;
import org.apache.hadoop.fs.FSError;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.LocalDirAllocator;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.permission.FsPermission;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.util.Shell;
import org.apache.hadoop.util.StringUtils;
import org.apache.log4j.Level;

/** Base class that runs a task in a separate process.  Tasks are run in a
 * separate process in order to isolate the map/reduce system code from bugs in
 * user supplied map and reduce functions.
 */
abstract class TaskRunner extends Thread {
  public static final Log LOG =
    LogFactory.getLog(TaskRunner.class);

  volatile boolean killed = false;
  private TaskTracker.TaskInProgress tip;
  private Task t;
  private Object lock = new Object();
  private volatile boolean done = false;
  private int exitCode = -1;
  private boolean exitCodeSet = false;
  
  private static String SYSTEM_PATH_SEPARATOR = System.getProperty("path.separator");

  
  private TaskTracker tracker;
  private TaskDistributedCacheManager taskDistributedCacheManager;

  protected JobConf conf;
  JvmManager jvmManager;

  public TaskRunner(TaskTracker.TaskInProgress tip, TaskTracker tracker, 
      JobConf conf) {
    this.tip = tip;
    this.t = tip.getTask();
    this.tracker = tracker;
    this.conf = conf;
    this.jvmManager = tracker.getJvmManagerInstance();
  }

  public Task getTask() { return t; }
  public TaskTracker.TaskInProgress getTaskInProgress() { return tip; }
  public TaskTracker getTracker() { return tracker; }

  public JvmManager getJvmManager() { return jvmManager; }

  /** Called when this task's output is no longer needed.
   * This method is run in the parent process after the child exits.  It should
   * not execute user code, only system code.
   */
  public void close() throws IOException {}


  /**
   * Get the java command line options for the child map/reduce tasks.
   * @param jobConf job configuration
   * @param defaultValue default value
   * @return the java command line options for child map/reduce tasks
   * @deprecated Use command line options specific to map or reduce tasks set 
   *             via {@link JobConf#MAPRED_MAP_TASK_JAVA_OPTS} or 
   *             {@link JobConf#MAPRED_REDUCE_TASK_JAVA_OPTS}
   */
  @Deprecated
  public String getChildJavaOpts(JobConf jobConf, String defaultValue) {
    return jobConf.get(JobConf.MAPRED_TASK_JAVA_OPTS, defaultValue);
  }
  
  /**
   * Get the maximum virtual memory of the child map/reduce tasks.
   * @param jobConf job configuration
   * @return the maximum virtual memory of the child task or <code>-1</code> if
   *         none is specified
   * @deprecated Use limits specific to the map or reduce tasks set via
   *             {@link JobConf#MAPRED_MAP_TASK_ULIMIT} or
   *             {@link JobConf#MAPRED_REDUCE_TASK_ULIMIT} 
   */
  @Deprecated
  public int getChildUlimit(JobConf jobConf) {
    return jobConf.getInt(JobConf.MAPRED_TASK_ULIMIT, -1);
  }
  
  /**
   * Get the environment variables for the child map/reduce tasks.
   * @param jobConf job configuration
   * @return the environment variables for the child map/reduce tasks or
   *         <code>null</code> if unspecified
   * @deprecated Use environment variables specific to the map or reduce tasks
   *             set via {@link JobConf#MAPRED_MAP_TASK_ENV} or
   *             {@link JobConf#MAPRED_REDUCE_TASK_ENV}
   */
  @Deprecated
  public String getChildEnv(JobConf jobConf) {
    return jobConf.get(JobConf.MAPRED_TASK_ENV);
  }

  /**
   * Get the log {@link Level} for the child map/reduce tasks.
   * @param jobConf
   * @return the log-level for the child map/reduce tasks
   */
  public abstract Level getLogLevel(JobConf jobConf);
  
  @Override
  public final void run() {
    String errorInfo = "Child Error";
    try {
      
      //before preparing the job localize 
      //all the archives
      TaskAttemptID taskid = t.getTaskID();
      final LocalDirAllocator lDirAlloc = new LocalDirAllocator(MRConfig.LOCAL_DIR);
      final File workDir = formWorkDir(lDirAlloc, taskid, t.isTaskCleanupTask(), conf);

      // We don't create any symlinks yet, so presence/absence of workDir
      // actually on the file system doesn't matter.
      tip.getUGI().doAs(new PrivilegedExceptionAction<Void>() {
        public Void run() throws IOException {
          taskDistributedCacheManager = 
            tracker.getTrackerDistributedCacheManager()
            .newTaskDistributedCacheManager(conf);
          taskDistributedCacheManager.setup(lDirAlloc, workDir, TaskTracker
              .getPrivateDistributedCacheDir(conf.getUser()), 
          TaskTracker.getPublicDistributedCacheDir());
          return null;
        }
      });

      // Set up the child task's configuration. After this call, no localization
      // of files should happen in the TaskTracker's process space. Any changes to
      // the conf object after this will NOT be reflected to the child.
      setupChildTaskConfiguration(lDirAlloc);

      // Build classpath
      List<String> classPaths =
          getClassPaths(conf, workDir, taskDistributedCacheManager);

      long logSize = TaskLog.getTaskLogLength(conf);

      //  Build exec child JVM args.
      Vector<String> vargs =
          getVMArgs(taskid, workDir, classPaths, logSize);

      tracker.addToMemoryManager(t.getTaskID(), t.isMapTask(), conf);

      // set memory limit using ulimit if feasible and necessary ...
      List<String> setup = getVMSetupCmd();

      // Set up the redirection of the task's stdout and stderr streams
      File[] logFiles = prepareLogFiles(taskid, t.isTaskCleanupTask());
      File stdout = logFiles[0];
      File stderr = logFiles[1];
      tracker.getTaskTrackerInstrumentation().reportTaskLaunch(taskid, stdout,
          stderr);

      Map<String, String> env = new HashMap<String, String>();
      errorInfo = getVMEnvironment(errorInfo, workDir, conf, env,
                                   taskid, logSize);

      launchJvmAndWait(setup, vargs, stdout, stderr, logSize, workDir, env);
      tracker.getTaskTrackerInstrumentation().reportTaskEnd(t.getTaskID());
      if (exitCodeSet) {
        if (!killed && exitCode != 0) {
          if (exitCode == 65) {
            tracker.getTaskTrackerInstrumentation().taskFailedPing(t.getTaskID());
          }
          throw new IOException("Task process exit with nonzero status of " +
              exitCode + ".");
        }
      }
    } catch (FSError e) {
      LOG.fatal("FSError", e);
      try {
        tracker.internalFsError(t.getTaskID(), e.getMessage());
      } catch (IOException ie) {
        LOG.fatal(t.getTaskID()+" reporting FSError", ie);
      }
    } catch (Throwable throwable) {
      LOG.warn(t.getTaskID() + " : " + errorInfo, throwable);
      Throwable causeThrowable = new Throwable(errorInfo, throwable);
      ByteArrayOutputStream baos = new ByteArrayOutputStream();
      causeThrowable.printStackTrace(new PrintStream(baos));
      try {
        tracker.internalReportDiagnosticInfo(t.getTaskID(), baos.toString());
      } catch (IOException e) {
        LOG.warn(t.getTaskID()+" Reporting Diagnostics", e);
      }
    } finally {
      try{
        if (taskDistributedCacheManager != null) {
          taskDistributedCacheManager.release();
        }
      }catch(IOException ie){
        LOG.warn("Error releasing caches : Cache files might not have been cleaned up");
      }

      // It is safe to call TaskTracker.TaskInProgress.reportTaskFinished with
      // *false* since the task has either
      // a) SUCCEEDED - which means commit has been done
      // b) FAILED - which means we do not need to commit
      tip.reportTaskFinished(false);
    }
  }

  void launchJvmAndWait(List<String> setup, Vector<String> vargs, File stdout,
      File stderr, long logSize, File workDir, Map<String, String> env)
      throws InterruptedException {
    jvmManager.launchJvm(this, jvmManager.constructJvmEnv(setup, vargs, stdout,
        stderr, logSize, workDir, env, conf));
    synchronized (lock) {
      while (!done) {
        lock.wait();
      }
    }
  }

  /**
   * Prepare the log files for the task
   * 
   * @param taskid
   * @param isCleanup
   * @return an array of files. The first file is stdout, the second is stderr.
   * @throws IOException 
   */
  File[] prepareLogFiles(TaskAttemptID taskid, boolean isCleanup)
      throws IOException {
    File[] logFiles = new File[2];
    logFiles[0] = TaskLog.getTaskLogFile(taskid, isCleanup,
        TaskLog.LogName.STDOUT);
    logFiles[1] = TaskLog.getTaskLogFile(taskid, isCleanup,
        TaskLog.LogName.STDERR);
    File logDir = logFiles[0].getParentFile();
    boolean b = logDir.mkdirs();
    if (!b) {
      LOG.warn("mkdirs failed. Ignoring");
    } else {
      FileSystem localFs = FileSystem.getLocal(conf);
      localFs.setPermission(new Path(logDir.getCanonicalPath()),
                            new FsPermission((short)0700));
    }

    return logFiles;
  }

  /**
   * Write the child's configuration to the disk and set it in configuration so
   * that the child can pick it up from there.
   * 
   * @param lDirAlloc
   * @throws IOException
   */
  void setupChildTaskConfiguration(LocalDirAllocator lDirAlloc)
      throws IOException {

    Path localTaskFile =
        lDirAlloc.getLocalPathForWrite(TaskTracker.getTaskConfFile(
            t.getUser(), t.getJobID().toString(), t.getTaskID().toString(), t
                .isTaskCleanupTask()), conf);

    // write the child's task configuration file to the local disk
    writeLocalTaskFile(localTaskFile.toString(), conf);

    // Set the final job file in the task. The child needs to know the correct
    // path to job.xml. So set this path accordingly.
    t.setJobFile(localTaskFile.toString());
  }

  /**
   * @return
   */
  private List<String> getVMSetupCmd() {

    int ulimit = getChildUlimit(conf);
    if (ulimit <= 0) {
      return null;
    }
    List<String> setup = null;
    String[] ulimitCmd = Shell.getUlimitMemoryCommand(ulimit);
    if (ulimitCmd != null) {
      setup = new ArrayList<String>();
      for (String arg : ulimitCmd) {
        setup.add(arg);
      }
    }
    return setup;
  }

  /**
   * Parse the given string and return an array of individual java opts. Split
   * on whitespace and replace the special string "@taskid@" with the task ID
   * given.
   * 
   * @param javaOpts The string to parse
   * @param taskid The task ID to replace the special string with
   * @return An array of individual java opts.
   */
  static String[] parseChildJavaOpts(String javaOpts, TaskAttemptID taskid) {
    javaOpts = javaOpts.replace("@taskid@", taskid.toString());
    return javaOpts.trim().split("\\s+");
  }

  /**
   * @param taskid
   * @param workDir
   * @param classPaths
   * @param logSize
   * @return
   * @throws IOException
   */
  private Vector<String> getVMArgs(TaskAttemptID taskid, File workDir,
      List<String> classPaths, long logSize)
      throws IOException {
    Vector<String> vargs = new Vector<String>(8);
    File jvm =                                  // use same jvm as parent
      new File(new File(System.getProperty("java.home"), "bin"), "java");

    vargs.add(jvm.toString());

    // Add child (task) java-vm options.
    //
    // The following symbols if present in mapred.{map|reduce}.child.java.opts 
    // value are replaced:
    // + @taskid@ is interpolated with value of TaskID.
    // Other occurrences of @ will not be altered.
    //
    // Example with multiple arguments and substitutions, showing
    // jvm GC logging, and start of a passwordless JVM JMX agent so can
    // connect with jconsole and the likes to watch child memory, threads
    // and get thread dumps.
    //
    //  <property>
    //    <name>mapred.child.java.opts</name>
    //    <value>-Xmx512M -verbose:gc -Xloggc:/tmp/@taskid@.gc \
    //           -Dcom.sun.management.jmxremote.authenticate=false \
    //           -Dcom.sun.management.jmxremote.ssl=false \
    //    </value>
    //  </property>
    //
    //  <property>
    //    <name>mapred.child.java.opts</name>
    //    <value>-Xmx1024M -verbose:gc -Xloggc:/tmp/@taskid@.gc \
    //           -Dcom.sun.management.jmxremote.authenticate=false \
    //           -Dcom.sun.management.jmxremote.ssl=false \
    //    </value>
    //  </property>
    //
    String[] javaOptsSplit = parseChildJavaOpts(getChildJavaOpts(conf,
                                       JobConf.DEFAULT_MAPRED_TASK_JAVA_OPTS),
                                       taskid);
    
    // Add java.library.path; necessary for loading native libraries.
    //
    // 1. To support native-hadoop library i.e. libhadoop.so, we add the 
    //    parent processes' java.library.path to the child. 
    // 2. We also add the 'cwd' of the task to it's java.library.path to help 
    //    users distribute native libraries via the DistributedCache.
    // 3. The user can also specify extra paths to be added to the 
    //    java.library.path via mapred.{map|reduce}.child.java.opts.
    //
    String libraryPath = System.getProperty("java.library.path");
    if (libraryPath == null) {
      libraryPath = workDir.getAbsolutePath();
    } else {
      libraryPath += SYSTEM_PATH_SEPARATOR + workDir;
    }
    boolean hasUserLDPath = false;
    for(int i=0; i<javaOptsSplit.length ;i++) { 
      if(javaOptsSplit[i].startsWith("-Djava.library.path=")) {
        javaOptsSplit[i] += SYSTEM_PATH_SEPARATOR + libraryPath;
        hasUserLDPath = true;
        break;
      }
    }
    if(!hasUserLDPath) {
      vargs.add("-Djava.library.path=" + libraryPath);
    }
    for (int i = 0; i < javaOptsSplit.length; i++) {
      vargs.add(javaOptsSplit[i]);
    }

    Path childTmpDir = createChildTmpDir(workDir, conf);
    vargs.add("-Djava.io.tmpdir=" + childTmpDir);

    // Add classpath.
    vargs.add("-classpath");
    String classPath = StringUtils.join(SYSTEM_PATH_SEPARATOR, classPaths);
    vargs.add(classPath);

    // Setup the log4j prop
    setupLog4jProperties(vargs, taskid, logSize);

    if (conf.getProfileEnabled()) {
      if (conf.getProfileTaskRange(t.isMapTask()
                                   ).isIncluded(t.getPartition())) {
        File prof = TaskLog.getTaskLogFile(taskid, t.isTaskCleanupTask(),
            TaskLog.LogName.PROFILE);
        vargs.add(String.format(conf.getProfileParams(), prof.toString()));
      }
    }

    // Add main class and its arguments 
    vargs.add(Child.class.getName());  // main of Child
    // pass umbilical address
    InetSocketAddress address = tracker.getTaskTrackerReportAddress();
    vargs.add(address.getAddress().getHostAddress()); 
    vargs.add(Integer.toString(address.getPort())); 
    vargs.add(taskid.toString());                      // pass task identifier
    // pass task log location
    vargs.add(TaskLog.getAttemptDir(taskid, t.isTaskCleanupTask()).toString());
    return vargs;
  }

  private void setupLog4jProperties(Vector<String> vargs, TaskAttemptID taskid,
      long logSize) {
    vargs.add("-Dhadoop.log.dir=" + 
        new File(System.getProperty("hadoop.log.dir")).getAbsolutePath());
    vargs.add("-Dhadoop.root.logger=" + getLogLevel(conf).toString() + ",TLA");
    vargs.add("-D" + TaskLogAppender.TASKID_PROPERTY +  "=" + taskid);
    vargs.add("-D" + TaskLogAppender.ISCLEANUP_PROPERTY +
              "=" + t.isTaskCleanupTask());
    vargs.add("-D" + TaskLogAppender.LOGSIZE_PROPERTY + "=" + logSize);
  }

  /**
   * @param taskid
   * @param workDir
   * @return
   * @throws IOException
   */
  static Path createChildTmpDir(File workDir,
      JobConf conf)
      throws IOException {

    // add java.io.tmpdir given by mapreduce.task.tmp.dir
    String tmp = conf.get(MRJobConfig.TASK_TEMP_DIR, "./tmp");
    Path tmpDir = new Path(tmp);

    // if temp directory path is not absolute, prepend it with workDir.
    if (!tmpDir.isAbsolute()) {
      tmpDir = new Path(workDir.toString(), tmp);

      FileSystem localFs = FileSystem.getLocal(conf);
      if (!localFs.mkdirs(tmpDir) && localFs.getFileStatus(tmpDir).isFile()) {
        throw new IOException("Mkdirs failed to create " + tmpDir.toString());
      }
    }
    return tmpDir;
  }

  /**
   */
  private static List<String> getClassPaths(JobConf conf, File workDir,
      TaskDistributedCacheManager taskDistributedCacheManager)
      throws IOException {
    // Accumulates class paths for child.
    List<String> classPaths = new ArrayList<String>();
    // start with same classpath as parent process
    appendSystemClasspaths(classPaths);

    // include the user specified classpath
    appendJobJarClasspaths(conf.getJar(), classPaths);
    
    // Distributed cache paths
    classPaths.addAll(taskDistributedCacheManager.getClassPaths());
    
    // Include the working dir too
    classPaths.add(workDir.toString());
    return classPaths;
  }

  /**
   * sets the environment variables needed for task jvm and its children.
   * @param errorInfo
   * @param workDir
   * @param env
   * @return
   * @throws Throwable
   */
  private String getVMEnvironment(String errorInfo, File workDir, JobConf conf,
      Map<String, String> env, TaskAttemptID taskid, long logSize)
      throws Throwable {
    StringBuffer ldLibraryPath = new StringBuffer();
    ldLibraryPath.append(workDir.toString());
    String oldLdLibraryPath = null;
    oldLdLibraryPath = System.getenv("LD_LIBRARY_PATH");
    if (oldLdLibraryPath != null) {
      ldLibraryPath.append(SYSTEM_PATH_SEPARATOR);
      ldLibraryPath.append(oldLdLibraryPath);
    }
    env.put("LD_LIBRARY_PATH", ldLibraryPath.toString());
    
    // put jobTokenFile name into env
    String jobTokenFile = conf.get(TokenCache.JOB_TOKENS_FILENAME);
    LOG.debug("putting jobToken file name into environment fn=" + jobTokenFile);
    env.put(UserGroupInformation.HADOOP_TOKEN_FILE_LOCATION, jobTokenFile);
    
    // for the child of task jvm, set hadoop.root.logger
    env.put("HADOOP_ROOT_LOGGER", "INFO,TLA");
    String hadoopClientOpts = System.getenv("HADOOP_CLIENT_OPTS");
    if (hadoopClientOpts == null) {
      hadoopClientOpts = "";
    } else {
      hadoopClientOpts = hadoopClientOpts + " ";
    }
    hadoopClientOpts = hadoopClientOpts + "-Dhadoop.tasklog.taskid=" + taskid
                       + " -Dhadoop.tasklog.iscleanup=" + t.isTaskCleanupTask()
                       + " -Dhadoop.tasklog.totalLogFileSize=" + logSize;
    env.put("HADOOP_CLIENT_OPTS", hadoopClientOpts);

    // add the env variables passed by the user
    String mapredChildEnv = getChildEnv(conf);
    if (mapredChildEnv != null && mapredChildEnv.length() > 0) {
      String childEnvs[] = mapredChildEnv.split(",");
      for (String cEnv : childEnvs) {
        try {
          String[] parts = cEnv.split("="); // split on '='
          String value = env.get(parts[0]);
          if (value != null) {
            // replace $env with the child's env constructed by tt's
            // example LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp
            value = parts[1].replace("$" + parts[0], value);
          } else {
            // this key is not configured by the tt for the child .. get it 
            // from the tt's env
            // example PATH=$PATH:/tmp
            value = System.getenv(parts[0]);
            if (value != null) {
              // the env key is present in the tt's env
              value = parts[1].replace("$" + parts[0], value);
            } else {
              // the env key is note present anywhere .. simply set it
              // example X=$X:/tmp or X=/tmp
              value = parts[1].replace("$" + parts[0], "");
            }
          }
          env.put(parts[0], value);
        } catch (Throwable t) {
          // set the error msg
          errorInfo = "Invalid User environment settings : " + mapredChildEnv 
                      + ". Failed to parse user-passed environment param."
                      + " Expecting : env1=value1,env2=value2...";
          LOG.warn(errorInfo);
          throw t;
        }
      }
    }
    return errorInfo;
  }

  /**
   * Write the task specific job-configuration file.
   * 
   * @param localFs
   * @throws IOException
   */
  private static void writeLocalTaskFile(String jobFile, JobConf conf)
      throws IOException {
    Path localTaskFile = new Path(jobFile);
    FileSystem localFs = FileSystem.getLocal(conf);
    localFs.delete(localTaskFile, true);
    OutputStream out = localFs.create(localTaskFile);
    try {
      conf.writeXml(out);
    } finally {
      out.close();
    }
  }

  /**
   * Prepare the Configs.LOCAL_DIR for the child. The child is sand-boxed now.
   * Whenever it uses LocalDirAllocator from now on inside the child, it will
   * only see files inside the attempt-directory. This is done in the Child's
   * process space.
   */
  static void setupChildMapredLocalDirs(Task t, JobConf conf) {
    String[] localDirs = conf.getTrimmedStrings(MRConfig.LOCAL_DIR);
    String jobId = t.getJobID().toString();
    String taskId = t.getTaskID().toString();
    boolean isCleanup = t.isTaskCleanupTask();
    String user = t.getUser();
    StringBuffer childMapredLocalDir =
        new StringBuffer(localDirs[0] + Path.SEPARATOR
            + TaskTracker.getLocalTaskDir(user, jobId, taskId, isCleanup));
    for (int i = 1; i < localDirs.length; i++) {
      childMapredLocalDir.append("," + localDirs[i] + Path.SEPARATOR
          + TaskTracker.getLocalTaskDir(user, jobId, taskId, isCleanup));
    }
    LOG.debug(MRConfig.LOCAL_DIR + " for child : " + childMapredLocalDir);
    conf.set(MRConfig.LOCAL_DIR, childMapredLocalDir.toString());
  }

  /** Creates the working directory pathname for a task attempt. */ 
  static File formWorkDir(LocalDirAllocator lDirAlloc, 
      TaskAttemptID task, boolean isCleanup, JobConf conf) 
      throws IOException {
    Path workDir =
        lDirAlloc.getLocalPathToRead(TaskTracker.getTaskWorkDir(
            conf.getUser(), task.getJobID().toString(), task.toString(),
            isCleanup), conf);

    return new File(workDir.toString());
  }

  private static void appendSystemClasspaths(List<String> classPaths) {
    for (String c : System.getProperty("java.class.path").split(
        SYSTEM_PATH_SEPARATOR)) {
      classPaths.add(c);
    }
  }
  
  /**
   * Given a "jobJar" (typically retrieved via {@link Configuration.getJar()}),
   * appends classpath entries for it, as well as its lib/ and classes/
   * subdirectories.
   * 
   * @param jobJar Job jar from configuration
   * @param classPaths Accumulator for class paths
   */
  static void appendJobJarClasspaths(String jobJar, List<String> classPaths) {
    if (jobJar == null) {
      return;
      
    }
    File jobCacheDir = new File(new Path(jobJar).getParent().toString());
    
    // if jar exists, it into workDir
    File[] libs = new File(jobCacheDir, "lib").listFiles();
    if (libs != null) {
      for (File l : libs) {
        classPaths.add(l.toString());
      }
    }
    classPaths.add(new File(jobCacheDir, "classes").toString());
    classPaths.add(new File(jobCacheDir, "job.jar").toString());
  }
  
  /**
   * Creates distributed cache symlinks and tmp directory, as appropriate.
   * Note that when we setup the distributed
   * cache, we didn't create the symlinks. This is done on a per task basis
   * by the currently executing task.
   * 
   * @param conf The job configuration.
   * @param workDir Working directory, which is completely deleted.
   */
  public static void setupWorkDir(JobConf conf, File workDir) throws IOException {
    if (LOG.isDebugEnabled()) {
      LOG.debug("Fully deleting contents of " + workDir);
    }

    /** deletes only the contents of workDir leaving the directory empty. We
     * can't delete the workDir as it is the current working directory.
     */
    FileUtil.fullyDeleteContents(workDir);
    
    if (DistributedCache.getSymlink(conf)) {
      URI[] archives = DistributedCache.getCacheArchives(conf);
      URI[] files = DistributedCache.getCacheFiles(conf);
      Path[] localArchives = DistributedCache.getLocalCacheArchives(conf);
      Path[] localFiles = DistributedCache.getLocalCacheFiles(conf);
      if (archives != null) {
        for (int i = 0; i < archives.length; i++) {
          String link = archives[i].getFragment();
          String target = localArchives[i].toString();
          symlink(workDir, target, link);
        }
      }
      if (files != null) {
        for (int i = 0; i < files.length; i++) {
          String link = files[i].getFragment();
          String target = localFiles[i].toString();
          symlink(workDir, target, link);
        }
      }
    }
    
    // For streaming, create extra symlinks (for all the files
    // in the job cache dir) in the current working directory.
    // Note that this is only executed if the configuration 
    // points to a jar file.
    if (conf.getJar() != null) {
      File jobCacheDir = new File(
          new Path(conf.getJar()).getParent().toString());
      try{
        TrackerDistributedCacheManager.createAllSymlink(conf, jobCacheDir,
            workDir);
      } catch(IOException ie){
        // Do not exit even if symlinks have not been created.
        LOG.warn(StringUtils.stringifyException(ie));
      }
    }

    createChildTmpDir(workDir, conf);
  }

  /**
   * Utility method for creating a symlink and warning on errors.
   * 
   * If link is null, does nothing.
   */
  private static void symlink(File workDir, String target, String link)
      throws IOException {
    if (link != null) {
      link = workDir.toString() + Path.SEPARATOR + link;
      File flink = new File(link);
      if (!flink.exists()) {
        LOG.info(String.format("Creating symlink: %s <- %s", target, link));
        if (0 != FileUtil.symLink(target, link)) {
          LOG.warn(String.format("Failed to create symlink: %s <- %s", target, link));
        }
      }
    }
  }

  /**
   * Kill the child process
   */
  public void kill() {
    killed = true;
    jvmManager.taskKilled(this);
    signalDone();
  }
  public void signalDone() {
    synchronized (lock) {
      done = true;
      lock.notify();
    }
  }
  public void setExitCode(int exitCode) {
    this.exitCodeSet = true;
    this.exitCode = exitCode;
  }
}

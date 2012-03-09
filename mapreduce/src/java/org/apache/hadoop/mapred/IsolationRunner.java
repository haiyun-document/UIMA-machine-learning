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

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.security.UserGroupInformation;
import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalDirAllocator;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JvmTask;
import org.apache.hadoop.mapreduce.MRConfig;
import org.apache.hadoop.mapreduce.split.JobSplit.TaskSplitIndex;

/**
 * IsolationRunner is intended to facilitate debugging by re-running a specific
 * task, given left-over task files for a (typically failed) past job.  
 * Currently, it is limited to re-running map tasks.
 *
 * Users may coerce MapReduce to keep task files around by setting 
 * mapreduce.task.files.preserve.failedtasks.  See mapred_tutorial.xml for more documentation.
 */
@InterfaceAudience.Public
@InterfaceStability.Evolving
public class IsolationRunner {
  private static final Log LOG = 
    LogFactory.getLog(IsolationRunner.class.getName());

  static class FakeUmbilical implements TaskUmbilicalProtocol {

    public long getProtocolVersion(String protocol, long clientVersion) {
      return TaskUmbilicalProtocol.versionID;
    }
    
    public void done(TaskAttemptID taskid) throws IOException {
      LOG.info("Task " + taskid + " reporting done.");
    }

    public void fsError(TaskAttemptID taskId, String message) throws IOException {
      LOG.info("Task " + taskId + " reporting file system error: " + message);
    }

    public void shuffleError(TaskAttemptID taskId, String message) throws IOException {
      LOG.info("Task " + taskId + " reporting shuffle error: " + message);
    }

    public void fatalError(TaskAttemptID taskId, String msg) throws IOException {
      LOG.info("Task " + taskId + " reporting fatal error: " + msg);
    }

    public JvmTask getTask(JvmContext context) throws IOException {
      return null;
    }

    public boolean ping(TaskAttemptID taskid) throws IOException {
      return true;
    }

    public void commitPending(TaskAttemptID taskId, TaskStatus taskStatus) 
    throws IOException, InterruptedException {
      statusUpdate(taskId, taskStatus);
    }
    
    public boolean canCommit(TaskAttemptID taskid) throws IOException {
      return true;
    }
    
    public boolean statusUpdate(TaskAttemptID taskId, TaskStatus taskStatus) 
    throws IOException, InterruptedException {
      StringBuffer buf = new StringBuffer("Task ");
      buf.append(taskId);
      buf.append(" making progress to ");
      buf.append(taskStatus.getProgress());
      String state = taskStatus.getStateString();
      if (state != null) {
        buf.append(" and state of ");
        buf.append(state);
      }
      LOG.info(buf.toString());
      // ignore phase
      // ignore counters
      return true;
    }

    public void reportDiagnosticInfo(TaskAttemptID taskid, String trace) throws IOException {
      LOG.info("Task " + taskid + " has problem " + trace);
    }
    
    public MapTaskCompletionEventsUpdate getMapCompletionEvents(JobID jobId, 
        int fromEventId, int maxLocs, TaskAttemptID id) throws IOException {
      return new MapTaskCompletionEventsUpdate(TaskCompletionEvent.EMPTY_ARRAY, 
                                               false);
    }

    public void reportNextRecordRange(TaskAttemptID taskid, 
        SortedRanges.Range range) throws IOException {
      LOG.info("Task " + taskid + " reportedNextRecordRange " + range);
    }
  }
  
  private ClassLoader makeClassLoader(JobConf conf, 
                                             File workDir) throws IOException {
    List<String> classPaths = new ArrayList<String>();
    // Add jar clas files (includes lib/* and classes/*)
    String jar = conf.getJar();
    if (jar != null) {
      TaskRunner.appendJobJarClasspaths(conf.getJar(), classPaths);
    }
    // Add the workdir, too.
    classPaths.add(workDir.toString());
    // Note: TaskRunner.run() does more, including DistributedCache files.
    
    // Convert to URLs
    URL[] urls = new URL[classPaths.size()];
    for (int i = 0; i < classPaths.size(); ++i) {
      urls[i] = new File(classPaths.get(i)).toURL();
    }
    return new URLClassLoader(urls);
  }
  
  /**
   * Main method.
   */
  boolean run(String[] args) 
      throws ClassNotFoundException, IOException, InterruptedException {
    if (args.length < 1) {
      System.out.println("Usage: IsolationRunner <path>/job.xml " +
      		"<optional-user-name>");
      return false;
    }
    File jobFilename = new File(args[0]);
    if (!jobFilename.exists() || !jobFilename.isFile()) {
      System.out.println(jobFilename + " is not a valid job file.");
      return false;
    }
    String user;
    if (args.length > 1) {
      user = args[1];
    } else {
      user = UserGroupInformation.getCurrentUser().getShortUserName();
    }
    JobConf conf = new JobConf(new Path(jobFilename.toString()));
    conf.setUser(user);
    TaskAttemptID taskId = TaskAttemptID.forName(conf.get(JobContext.TASK_ATTEMPT_ID));
    if (taskId == null) {
      System.out.println("mapreduce.task.attempt.id not found in configuration;" + 
          " job.xml is not a task config");
    }
    boolean isMap = conf.getBoolean(JobContext.TASK_ISMAP, true);
    if (!isMap) {
      System.out.println("Only map tasks are supported.");
      return false;
    }
    int partition = conf.getInt(JobContext.TASK_PARTITION, 0);
    
    // setup the local and user working directories
    FileSystem local = FileSystem.getLocal(conf);
    LocalDirAllocator lDirAlloc = new LocalDirAllocator(MRConfig.LOCAL_DIR);

    File workDirName = TaskRunner.formWorkDir(lDirAlloc, taskId, false, conf);
    local.setWorkingDirectory(new Path(workDirName.toString()));
    FileSystem.get(conf).setWorkingDirectory(conf.getWorkingDirectory());
    
    // set up a classloader with the right classpath
    ClassLoader classLoader =
        makeClassLoader(conf, new File(workDirName.toString()));
    Thread.currentThread().setContextClassLoader(classLoader);
    conf.setClassLoader(classLoader);

    // split.dta/split.meta files are used only by IsolationRunner. 
    // The file can now be in any of the configured local disks, 
    // so use LocalDirAllocator to find out where it is.
    Path localMetaSplit =
        new LocalDirAllocator(MRConfig.LOCAL_DIR).getLocalPathToRead(
            TaskTracker.getLocalSplitMetaFile(conf.getUser(), 
              taskId.getJobID().toString(), taskId
                .toString()), conf);
    DataInputStream splitFile = FileSystem.getLocal(conf).open(localMetaSplit);
    TaskSplitIndex splitIndex = new TaskSplitIndex(); 
    splitIndex.readFields(splitFile);
    splitFile.close();

    Task task = 
      new MapTask(jobFilename.toString(), taskId, partition, splitIndex, 1);
    task.setConf(conf);
    task.run(conf, new FakeUmbilical());
    return true;
  }


  /**
   * Run a single task.
   *
   * @param args the first argument is the task directory
   */
  public static void main(String[] args) 
      throws ClassNotFoundException, IOException, InterruptedException {
    if (!new IsolationRunner().run(args)) {
      System.exit(1);
    }
  }
}

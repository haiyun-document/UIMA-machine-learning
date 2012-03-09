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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.mapred.JobQueueJobInProgressListener.JobSchedulingInfo;
import org.apache.hadoop.util.StringUtils;

/**
 * This class asynchronously initializes jobs submitted to the
 * Map/Reduce cluster running with the {@link CapacityTaskScheduler}.
 *
 * <p>
 * The class comprises of a main poller thread, and a set of worker
 * threads that together initialize the jobs. The poller thread periodically
 * looks at jobs submitted to the scheduler, and selects a set of them
 * to be initialized. It passes these to the worker threads for initializing.
 * Each worker thread is configured to look at jobs submitted to a fixed
 * set of queues. It initializes jobs in a round robin manner - selecting
 * the first job in order from each queue ready to be initialized.
 * </p>
 * 
 * <p>
 * An initialized job occupies memory resources on the Job Tracker. Hence,
 * the poller limits the number of jobs initialized at any given time to
 * a configured limit. The limit is specified per user per queue.
 * </p>
 * 
 * <p>
 * However, since a job needs to be initialized before the scheduler can
 * select tasks from it to run, it tries to keep a backlog of jobs 
 * initialized so the scheduler does not need to wait and let empty slots
 * go waste. The core logic of the poller is to pick up the right jobs,
 * which have a good potential to be run next by the scheduler. To do this,
 * it picks up jobs submitted across users and across queues to account
 * both for guaranteed capacities and user limits. It also always initializes
 * high priority jobs, whenever they need to be initialized, even if this
 * means going over the limit for initialized jobs.
 * </p>
 */
public class JobInitializationPoller extends Thread {

  private static final Log LOG = LogFactory
      .getLog(JobInitializationPoller.class.getName());

  /*
   * The poller picks up jobs across users to initialize based on user limits.
   * Suppose the user limit for a queue is 25%, it means atmost 4 users' jobs
   * can run together. However, in order to account for jobs from a user that
   * might complete faster than others, it initializes jobs from an additional
   * number of users as a backlog. This variable defines the additional
   * number of users whose jobs can be considered for initializing. 
   */
  private static final int MAX_ADDITIONAL_USERS_TO_INIT = 2;

  private JobQueuesManager jobQueueManager;
  private long sleepInterval;
  private int poolSize;

  /**
   * A worker thread that initializes jobs in one or more queues assigned to
   * it.
   *
   * Jobs are initialized in a round robin fashion one from each queue at a
   * time.
   */
  class JobInitializationThread extends Thread {

    private JobInProgress initializingJob;

    private volatile boolean startIniting;
    private AtomicInteger currentJobCount = new AtomicInteger(0); // number of jobs to initialize

    /**
     * The hash map which maintains relationship between queue to jobs to
     * initialize per queue.
     */
    private HashMap<String, TreeMap<JobSchedulingInfo, JobInProgress>> jobsPerQueue;

    public JobInitializationThread() {
      startIniting = true;
      jobsPerQueue = new HashMap<String, TreeMap<JobSchedulingInfo, JobInProgress>>();
    }

    @Override
    public void run() {
      while (startIniting) {
        initializeJobs();  
        try {
          if (startIniting) {
            Thread.sleep(sleepInterval);
          } else {
            break;
          }
        } catch (Throwable t) {
        }
      }
    }

    // The key method that initializes jobs from queues
    // This method is package-private to allow test cases to call it
    // synchronously in a controlled manner.
    void initializeJobs() {
      // while there are more jobs to initialize...
      while (currentJobCount.get() > 0) {
        Set<String> queues = jobsPerQueue.keySet();
        for (String queue : queues) {
          JobInProgress job = getFirstJobInQueue(queue);
          if (job == null) {
            continue;
          }
          LOG.info("Initializing job : " + job.getJobID() + " in AbstractQueue "
              + job.getProfile().getQueueName() + " For user : "
              + job.getProfile().getUser());
          if (startIniting) {
            setInitializingJob(job);
            ttm.initJob(job);
            setInitializingJob(null);
          } else {
            break;
          }
        }
      }
    }

    /**
     * This method returns the first job in the queue and removes the same.
     * 
     * @param queue
     *          queue name
     * @return First job in the queue and removes it.
     */
    private JobInProgress getFirstJobInQueue(String queue) {
      TreeMap<JobSchedulingInfo, JobInProgress> jobsList = jobsPerQueue
          .get(queue);
      synchronized (jobsList) {
        if (jobsList.isEmpty()) {
          return null;
        }
        Iterator<JobInProgress> jobIterator = jobsList.values().iterator();
        JobInProgress job = jobIterator.next();
        jobIterator.remove();
        currentJobCount.getAndDecrement();
        return job;
      }
    }

    /*
     * Test method to check if the thread is currently initialising the job
     */
    synchronized JobInProgress getInitializingJob() {
      return this.initializingJob;
    }
    
    synchronized void setInitializingJob(JobInProgress job) {
      this.initializingJob  = job;
    }

    void terminate() {
      startIniting = false;
    }

    void addJobsToQueue(String queue, JobInProgress job) {
      TreeMap<JobSchedulingInfo, JobInProgress> jobs = jobsPerQueue
          .get(queue);
      if (jobs == null) {
        LOG.error("Invalid queue passed to the thread : " + queue
            + " For job :: " + job.getJobID());
        return;
      }
      synchronized (jobs) {
        JobSchedulingInfo schedInfo = new JobSchedulingInfo(job);
        jobs.put(schedInfo, job);
        currentJobCount.getAndIncrement();
      }
    }

    void addQueue(String queue) {
      TreeMap<JobSchedulingInfo, JobInProgress> jobs = new TreeMap<JobSchedulingInfo, JobInProgress>(
          jobQueueManager.getComparator(queue));
      jobsPerQueue.put(queue, jobs);
    }
  }

  /**
   * The queue information class maintains following information per queue:
   * Maximum users allowed to initialize job in the particular queue. Maximum
   * jobs allowed to be initialize per user in the queue.
   * 
   */
  private static class QueueInfo {
    int maxUsersAllowedToInitialize;
    int maxJobsPerUserToInitialize;

    public QueueInfo(int maxUsersAllowedToInitialize,
        int maxJobsPerUserToInitialize) {
      this.maxJobsPerUserToInitialize = maxJobsPerUserToInitialize;
      this.maxUsersAllowedToInitialize = maxUsersAllowedToInitialize;
    }
  }

  /**
   * Map which contains the configuration used for initializing jobs
   * in that associated to a particular job queue.
   */
  private HashMap<String, QueueInfo> jobQueues;

  /**
   * Set of jobs which have been passed to Initialization threads.
   * This is maintained so that we dont call initTasks() for same job twice.
   */
  private HashMap<JobID,JobInProgress> initializedJobs;

  private volatile boolean running;

  private TaskTrackerManager ttm;
  /**
   * The map which provides information which thread should be used to
   * initialize jobs for a given job queue.
   */
  private HashMap<String, JobInitializationThread> threadsToQueueMap;

  public JobInitializationPoller(
    JobQueuesManager mgr,
    TaskTrackerManager ttm) {
    initializedJobs = new HashMap<JobID,JobInProgress>();
    jobQueues = new HashMap<String, QueueInfo>();
    this.jobQueueManager = mgr;
    threadsToQueueMap = new HashMap<String, JobInitializationThread>();
    super.setName("JobInitializationPollerThread");
    running = true;
    this.ttm = ttm;
  }

  /*
   * method to read all configuration values required by the initialisation
   * poller
   */

  void init(Set<String> queues, 
            CapacitySchedulerConf capacityConf) {
    setupJobInitializerConfiguration(queues, capacityConf);
    assignThreadsToQueues();
    Collection<JobInitializationThread> threads = threadsToQueueMap.values();
    for (JobInitializationThread t : threads) {
      if (!t.isAlive()) {
        t.setDaemon(true);
        t.start();
      }
    }
  }

  /**
   * Initialize the configuration of the JobInitializer as well as of the specific
   * queues.
   * 
   * @param queues
   * @param schedulerConf
   */
  private void setupJobInitializerConfiguration(Set<String> queues,
      CapacitySchedulerConf schedulerConf) {
    for (String queue : queues) {
      int maxUsersToInitialize = getMaxUsersToInit(schedulerConf, queue);
      int maxJobsPerUserToInitialize =
          schedulerConf.getMaxJobsPerUserToInitialize(queue);
      QueueInfo qi =
          new QueueInfo(maxUsersToInitialize, maxJobsPerUserToInitialize);
      jobQueues.put(queue, qi);
    }
    sleepInterval = schedulerConf.getSleepInterval();
    poolSize = schedulerConf.getMaxWorkerThreads();
    if (poolSize > queues.size()) {
      poolSize = queues.size();
    }
  }

  /**
   * 
   * @param schedulerConf
   * @param queue
   * @return
   */
  private int getMaxUsersToInit(CapacitySchedulerConf schedulerConf,
      String queue) {
    int userlimit = schedulerConf.getMinimumUserLimitPercent(queue);
    return (100 / userlimit) + MAX_ADDITIONAL_USERS_TO_INIT;
  }

  /**
   * Refresh the Scheduler configuration cached with the initializer. This
   * should be called only by
   * {@link CapacityTaskScheduler.CapacitySchedulerQueueRefresher#refreshQueues()}
   * . The cached configuration currently is only used by the main thread in the
   * initializer. So, any updates are picked up automatically by subsequent
   * iterations of the main thread.
   */
  void refreshQueueInfo(CapacitySchedulerConf schedulerConf) {
    for (String queue : jobQueues.keySet()) {
      QueueInfo queueInfo = jobQueues.get(queue);
      synchronized (queueInfo) {
        queueInfo.maxUsersAllowedToInitialize =
            getMaxUsersToInit(schedulerConf, queue);
        queueInfo.maxJobsPerUserToInitialize =
            schedulerConf.getMaxJobsPerUserToInitialize(queue);
      }
    }
  }

  /**
   * This is main thread of initialization poller, We essentially do 
   * following in the main threads:
   * 
   * <ol>
   * <li> Clean up the list of initialized jobs list which poller maintains
   * </li>
   * <li> Select jobs to initialize in the polling interval.</li>
   * </ol>
   */
  public void run() {
    while (running) {
      try {
        cleanUpInitializedJobsList();
        selectJobsToInitialize();
        if (!this.isInterrupted()) {
          Thread.sleep(sleepInterval);
        }
      } catch (InterruptedException e) {
        LOG.error("Job Initialization poller interrupted"
            + StringUtils.stringifyException(e));
      }
    }
  }

  /**
   * The key method which does selecting jobs to be initalized across 
   * queues and assign those jobs to their appropriate init-worker threads.
   * <br/>
   * This method is overriden in test case which is used to test job
   * initialization poller.
   * 
   */
  void selectJobsToInitialize() {
    for (String queue : jobQueues.keySet()) {
      ArrayList<JobInProgress> jobsToInitialize = getJobsToInitialize(queue);
      printJobs(jobsToInitialize);
      JobInitializationThread t = threadsToQueueMap.get(queue);
      for (JobInProgress job : jobsToInitialize) {
        t.addJobsToQueue(queue, job);
      }
    }
  }

  /**
   * Method used to print log statements about which jobs are being
   * passed to init-threads. 
   * 
   * @param jobsToInitialize list of jobs which are passed to be 
   * init-threads.
   */
  private void printJobs(ArrayList<JobInProgress> jobsToInitialize) {
    for (JobInProgress job : jobsToInitialize) {
      LOG.info("Passing to Initializer Job Id :" + job.getJobID()
          + " User: " + job.getProfile().getUser() + " AbstractQueue : "
          + job.getProfile().getQueueName());
    }
  }

  /**
   * This method exists to be overridden by test cases that wish to
   * create a test-friendly worker thread which can be controlled
   * synchronously.
   * 
   * @return Instance of worker init-threads.
   */
  JobInitializationThread createJobInitializationThread() {
    return new JobInitializationThread();
  }
  
  /**
   * Method which is used by the poller to assign appropriate worker thread
   * to a queue. The number of threads would be always less than or equal
   * to number of queues in a system. If number of threads is configured to 
   * be more than number of queues then poller does not create threads more
   * than number of queues. 
   * 
   */
  private void assignThreadsToQueues() {
    int countOfQueues = jobQueues.size();
    String[] queues = (String[]) jobQueues.keySet().toArray(
        new String[countOfQueues]);
    int numberOfQueuesPerThread = countOfQueues / poolSize;
    int numberOfQueuesAssigned = 0;
    for (int i = 0; i < poolSize; i++) {
      JobInitializationThread initializer = createJobInitializationThread();
      int batch = (i * numberOfQueuesPerThread);
      for (int j = batch; j < (batch + numberOfQueuesPerThread); j++) {
        initializer.addQueue(queues[j]);
        threadsToQueueMap.put(queues[j], initializer);
        numberOfQueuesAssigned++;
      }
    }

    if (numberOfQueuesAssigned < countOfQueues) {
      // Assign remaining queues in round robin fashion to other queues
      int startIndex = 0;
      for (int i = numberOfQueuesAssigned; i < countOfQueues; i++) {
        JobInitializationThread t = threadsToQueueMap
            .get(queues[startIndex]);
        t.addQueue(queues[i]);
        threadsToQueueMap.put(queues[i], t);
        startIndex++;
      }
    }
  }

  /**
   * 
   * Method used to select jobs to be initialized for a given queue. <br/>
   * 
   * We want to ensure that enough jobs have been initialized, so that when the
   * Scheduler wants to consider a new job to run, it's ready. We clearly don't
   * want to initialize too many jobs as each initialized job has a memory
   * footprint, sometimes significant.
   * 
   * Number of jobs to be initialized is restricted by two values: - Maximum
   * number of users whose jobs we want to initialize, which is equal to 
   * the number of concurrent users the queue can support. - Maximum number 
   * of initialized jobs per user. The product of these two gives us the
   * total number of initialized jobs.
   * 
   * Note that this is a rough number, meant for decreasing extra memory
   * footprint. It's OK if we go over it once in a while, if we have to.
   * 
   * This can happen as follows. Suppose we have initialized 3 jobs for a
   * user. Now, suppose the user submits a job who's priority is higher than
   * that of the 3 jobs initialized. This job needs to be initialized, since it
   * will run earlier than the 3 jobs. We'll now have 4 initialized jobs for the
   * user. If memory becomes a problem, we should ideally un-initialize one of
   * the 3 jobs, to keep the count of initialized jobs at 3, but that's
   * something we don't do for now. This situation can also arise when a new
   * user submits a high priority job, thus superceeding a user whose jobs have
   * already been initialized. The latter user's initialized jobs are redundant,
   * but we'll leave them initialized.
   * 
   * @param queue name of the queue to pick the jobs to initialize.
   * @return list of jobs to be initalized in a queue. An empty queue is
   *         returned if no jobs are found.
   */
  ArrayList<JobInProgress> getJobsToInitialize(String queue) {
    QueueInfo qi = jobQueues.get(queue);
    ArrayList<JobInProgress> jobsToInitialize = new ArrayList<JobInProgress>();
    // use the configuration parameter which is configured for the particular
    // queue.
    int maximumUsersAllowedToInitialize;
    int maxJobsPerUserAllowedToInitialize;
    synchronized (qi) {
      maximumUsersAllowedToInitialize = qi.maxUsersAllowedToInitialize;
      maxJobsPerUserAllowedToInitialize = qi.maxJobsPerUserToInitialize;
    }
    int maxJobsPerQueueToInitialize = maximumUsersAllowedToInitialize
        * maxJobsPerUserAllowedToInitialize;
    int countOfJobsInitialized = 0;
    HashMap<String, Integer> userJobsInitialized = new HashMap<String, Integer>();
    Collection<JobInProgress> jobs = jobQueueManager.getJobQueue(queue).getWaitingJobs();
    /*
     * Walk through the collection of waiting jobs.
     *  We maintain a map of jobs that have already been initialized. If a 
     *  job exists in that map, increment the count for that job's user 
     *  and move on to the next job.
     *   
     *  If the job doesn't exist, see whether we  want to initialize it. 
     *  We initialize it if: - at least one job of the user has already 
     *  been initialized, but the user's total initialized jobs are below 
     *  the limit, OR - this is a new user, and we haven't reached the limit
     *  for the number of users whose jobs we want to initialize. We break 
     *  when we've reached the limit of maximum jobs to initialize.
     */
    for (JobInProgress job : jobs) {
      String user = job.getProfile().getUser();
      int numberOfJobs = userJobsInitialized.get(user) == null ? 0
          : userJobsInitialized.get(user);
      // If the job is already initialized then add the count against user
      // then continue.
      if (initializedJobs.containsKey(job.getJobID())) {
        userJobsInitialized.put(user, Integer.valueOf(numberOfJobs + 1));
        countOfJobsInitialized++;
        continue;
      }
      boolean isUserPresent = userJobsInitialized.containsKey(user);
      if (!isUserPresent
          && userJobsInitialized.size() < maximumUsersAllowedToInitialize) {
        // this is a new user being considered and the number of users
        // is within limits.
        userJobsInitialized.put(user, Integer.valueOf(numberOfJobs + 1));
        jobsToInitialize.add(job);
        initializedJobs.put(job.getJobID(),job);
        countOfJobsInitialized++;
      } else if (isUserPresent
          && numberOfJobs < maxJobsPerUserAllowedToInitialize) {
        userJobsInitialized.put(user, Integer.valueOf(numberOfJobs + 1));
        jobsToInitialize.add(job);
        initializedJobs.put(job.getJobID(),job);
        countOfJobsInitialized++;
      }
      /*
       * if the maximum number of jobs to initalize for a queue is reached
       * then we stop looking at further jobs. The jobs beyond this number
       * can be initialized.
       */
      if(countOfJobsInitialized > maxJobsPerQueueToInitialize) {
        break;
      }
    }
    return jobsToInitialize;
  }


  /**
   * Method which is used internally to clean up the initialized jobs
   * data structure which the job initialization poller uses to check
   * if a job is initalized or not.
   * 
   * Algorithm for cleaning up task is as follows:
   * 
   * <ul>
   * <li> For jobs in <b>initalizedJobs</b> list </li>
   * <ul>
   * <li> If job is running</li>
   * <ul>
   * <li> If job is scheduled then remove the job from the waiting queue 
   * of the scheduler and <b>initalizedJobs</b>.<br/>
   *  The check for a job is scheduled or not is done by following 
   *  formulae:<br/> 
   *  if pending <i>task</i> &lt; desired <i>task</i> then scheduled else
   *  not scheduled.<br/>
   *  The formulae would return <i>scheduled</i> if one task has run or failed,
   *  any cases in which there has been a failure but not enough to mark task 
   *  as failed, we return <i>not scheduled</i> in formulae.
   * </li>
   * </ul>
   * 
   * <li> If job is complete, then remove the job from <b>initalizedJobs</b>.
   * </li>
   * 
   * </ul>
   * </ul>
   * 
   */
  void cleanUpInitializedJobsList() {
    Iterator<Entry<JobID, JobInProgress>> jobsIterator = 
      initializedJobs.entrySet().iterator();
    while(jobsIterator.hasNext()) {
      Entry<JobID,JobInProgress> entry = jobsIterator.next();
      JobInProgress job = entry.getValue();
      if (job.getStatus().getRunState() == JobStatus.RUNNING) {
        if (isScheduled(job)) {
          LOG.info("Removing scheduled jobs from waiting queue"
              + job.getJobID());
          jobsIterator.remove();
          jobQueueManager.getJobQueue(job).removeWaitingJob(new JobSchedulingInfo(job));
          continue;
        }
      }
      if(job.isComplete()) {
        LOG.info("Removing killed/completed job from initalized jobs " +
        		"list : "+ job.getJobID());
        jobsIterator.remove();
      }
    }
  }

  /**
   * Convenience method to check if job has been scheduled or not.
   * 
   * The method may return false in case of job which has failure but
   * has not failed the tip.
   * @param job
   * @return
   */
  private boolean isScheduled(JobInProgress job) {
    return ((job.pendingMaps() < job.desiredMaps()) 
        || (job.pendingReduces() < job.desiredReduces()));
  }

  void terminate() {
    running = false;
    for (Entry<String, JobInitializationThread> entry : threadsToQueueMap
        .entrySet()) {
      JobInitializationThread t = entry.getValue();
      if (t.isAlive()) {
        t.terminate();
        t.interrupt();
      }
    }
  }

  /*
   * Test method used only for testing purposes.
   */
  JobInProgress getInitializingJob(String queue) {
    JobInitializationThread t = threadsToQueueMap.get(queue);
    if (t == null) {
      return null;
    } else {
      return t.getInitializingJob();
    }
  }

  Set<JobID> getInitializedJobList() {
    return initializedJobs.keySet();
  }

  public HashMap<String, JobInitializationThread> getThreadsToQueueMap() {
    return threadsToQueueMap;
  }
  
  public long getSleepInterval(){
    return sleepInterval;
  }
}

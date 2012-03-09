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

#include "fuse_dfs.h"
#include "fuse_impls.h"
#include "fuse_connect.h"

int dfs_utimens(const char *path, const struct timespec ts[2])
{
  TRACE1("utimens", path)
#if PERMS
  // retrieve dfs specific data
  dfs_context *dfs = (dfs_context*)fuse_get_context()->private_data;

  // check params and the context var
  assert(path);
  assert(dfs);
  assert('/' == *path);

  time_t aTime = ts[0].tv_sec;
  time_t mTime = ts[1].tv_sec;

  hdfsFS userFS;
  // if not connected, try to connect and fail out if we can't.
  if ((userFS = doConnectAsUser(dfs->nn_hostname,dfs->nn_port))== NULL) {
    syslog(LOG_ERR, "ERROR: could not connect to dfs %s:%d\n",
           __FILE__, __LINE__);
    return -EIO;
  }

  if (hdfsUtime(userFS, path, mTime, aTime)) {
    hdfsFileInfo *info = hdfsGetPathInfo(dfs->fs,path);
    if (info == NULL) {
      return -EIO;
    }
    // Silently ignore utimens failure for directories, otherwise 
    // some programs like tar will fail.
    if (info->mKind == kObjectKindDirectory) {
      return 0;
    }
    return -errno;
  }
#endif  
  return 0;
}

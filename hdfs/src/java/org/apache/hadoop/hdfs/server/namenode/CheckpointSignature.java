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
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.hdfs.server.common.StorageInfo;
import org.apache.hadoop.hdfs.server.namenode.FSImage;
import org.apache.hadoop.io.MD5Hash;
import org.apache.hadoop.io.WritableComparable;

/**
 * A unique signature intended to identify checkpoint transactions.
 */
@InterfaceAudience.Private
public class CheckpointSignature extends StorageInfo 
                      implements WritableComparable<CheckpointSignature> {
  private static final String FIELD_SEPARATOR = ":";
  long editsTime = -1L;
  long checkpointTime = -1L;
  MD5Hash imageDigest = null;

  public CheckpointSignature() {}

  CheckpointSignature(FSImage fsImage) {
    super(fsImage);
    editsTime = fsImage.getEditLog().getFsEditTime();
    checkpointTime = fsImage.getCheckpointTime();
    imageDigest = fsImage.imageDigest;
  }

  CheckpointSignature(String str) {
    String[] fields = str.split(FIELD_SEPARATOR);
    assert fields.length == 6 : "Must be 6 fields in CheckpointSignature";
    layoutVersion = Integer.valueOf(fields[0]);
    namespaceID = Integer.valueOf(fields[1]);
    cTime = Long.valueOf(fields[2]);
    editsTime = Long.valueOf(fields[3]);
    checkpointTime = Long.valueOf(fields[4]);
    imageDigest = new MD5Hash(fields[5]);
  }

  /**
   * Get the MD5 image digest
   * @return the MD5 image digest
   */
  MD5Hash getImageDigest() {
    return imageDigest;
  }

  public String toString() {
    return String.valueOf(layoutVersion) + FIELD_SEPARATOR
         + String.valueOf(namespaceID) + FIELD_SEPARATOR
         + String.valueOf(cTime) + FIELD_SEPARATOR
         + String.valueOf(editsTime) + FIELD_SEPARATOR
         + String.valueOf(checkpointTime) + FIELD_SEPARATOR
         +  imageDigest.toString();
  }

  void validateStorageInfo(FSImage si) throws IOException {
    if(layoutVersion != si.layoutVersion
        || namespaceID != si.namespaceID || cTime != si.cTime
        || checkpointTime != si.checkpointTime ||
        !imageDigest.equals(si.imageDigest)) {
      // checkpointTime can change when the image is saved - do not compare
      throw new IOException("Inconsistent checkpoint fields.\n"
          + "LV = " + layoutVersion + " namespaceID = " + namespaceID
          + " cTime = " + cTime + "; checkpointTime = " + checkpointTime
          + " ; imageDigest = " + imageDigest
          + ".\nExpecting respectively: "
          + si.layoutVersion + "; " + si.namespaceID + "; " + si.cTime
          + "; " + si.checkpointTime + "; " + si.imageDigest);
    }
  }

  //
  // Comparable interface
  //
  public int compareTo(CheckpointSignature o) {
    return 
      (layoutVersion < o.layoutVersion) ? -1 : 
                  (layoutVersion > o.layoutVersion) ? 1 :
      (namespaceID < o.namespaceID) ? -1 : (namespaceID > o.namespaceID) ? 1 :
      (cTime < o.cTime) ? -1 : (cTime > o.cTime) ? 1 :
      (editsTime < o.editsTime) ? -1 : (editsTime > o.editsTime) ? 1 :
      (checkpointTime < o.checkpointTime) ? -1 : 
                  (checkpointTime > o.checkpointTime) ? 1 :
                    imageDigest.compareTo(o.imageDigest);
  }

  public boolean equals(Object o) {
    if (!(o instanceof CheckpointSignature)) {
      return false;
    }
    return compareTo((CheckpointSignature)o) == 0;
  }

  public int hashCode() {
    return layoutVersion ^ namespaceID ^
            (int)(cTime ^ editsTime ^ checkpointTime) ^
            imageDigest.hashCode();
  }

  /////////////////////////////////////////////////
  // Writable
  /////////////////////////////////////////////////
  public void write(DataOutput out) throws IOException {
    super.write(out);
    out.writeLong(editsTime);
    out.writeLong(checkpointTime);
    imageDigest.write(out);
  }

  public void readFields(DataInput in) throws IOException {
    super.readFields(in);
    editsTime = in.readLong();
    checkpointTime = in.readLong();
    imageDigest = new MD5Hash();
    imageDigest.readFields(in);
  }
}

#ifdef USE_LMDB
#ifndef CAFFE_UTIL_DB_LMDB_HPP
#define CAFFE_UTIL_DB_LMDB_HPP

#include <string>
#include <vector>

#include "lmdb.h"

#include "caffe/util/db.hpp"

namespace caffe { namespace db {

inline void MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}

class LMDBCursor : public Cursor {
 public:
  explicit LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor)
    : mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false) {
    SeekToFirst();
  }
  virtual ~LMDBCursor() {
    mdb_cursor_close(mdb_cursor_);
    mdb_txn_abort(mdb_txn_);
  }
  virtual void SeekToFirst() { Seek(MDB_FIRST); }
  virtual void Next() { Seek(MDB_NEXT); }
  virtual string key() {
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
  virtual string value() {
    return string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  }
  virtual bool valid() { return valid_; }

 //private:
 protected:
  void Seek(MDB_cursor_op op) {
    int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      valid_ = false;
    } else {
      MDB_CHECK(mdb_status);
      valid_ = true;
    }
  }

  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  bool valid_;
};

class LMDBShuffleCursor : public LMDBCursor {
 public:
  explicit LMDBShuffleCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor)
    : LMDBCursor(mdb_txn, mdb_cursor) {
    GetKeys(&keys_);

    keys_permutation_.resize(keys_.size());
    for (int i = 0; i < keys_.size(); ++i) {
      keys_permutation_[i] = i;
    }

    SeekToFirst();
  }
  virtual ~LMDBShuffleCursor() {}
  virtual void SeekToFirst() { 
    std::random_shuffle(keys_permutation_.begin(), keys_permutation_.end());
    keys_iter_ = 0;
  }
  virtual void Next() { 
    ++keys_iter_;
    if (keys_iter_ >= keys_.size()) {
      valid_ = false;
    } else {
      mdb_key_ = keys_[keys_permutation_[keys_iter_]];
      Seek(MDB_SET_KEY);
    }
  }

 private:
  void GetKeys(vector<MDB_val>* keys) {
    LMDBCursor::SeekToFirst();
    keys->clear();
    while (valid()) {
      keys->push_back(mdb_key_);
      LMDBCursor::Next();
    }
  }

  vector<MDB_val> keys_;
  vector<int> keys_permutation_;
  int keys_iter_;    
};


class LMDBTransaction : public Transaction {
 public:
  explicit LMDBTransaction(MDB_env* mdb_env)
    : mdb_env_(mdb_env) { }
  virtual void Put(const string& key, const string& value);
  virtual void Commit();

 private:
  MDB_env* mdb_env_;
  vector<string> keys, values;

  void DoubleMapSize();

  DISABLE_COPY_AND_ASSIGN(LMDBTransaction);
};

class LMDB : public DB {
 public:
  LMDB() : mdb_env_(NULL) { }
  virtual ~LMDB() { Close(); }
  virtual void Open(const string& source, Mode mode);
  virtual void Close() {
    if (mdb_env_ != NULL) {
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
      mdb_env_ = NULL;
    }
  }
//  virtual LMDBCursor* NewCursor();
  virtual LMDBCursor* NewCursor(DataParameter::CURSOR_TYPE type);

  virtual LMDBTransaction* NewTransaction();

 private:
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
#endif  // USE_LMDB

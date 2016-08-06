import numpy
import csv
import tensorflow as tf
import pickle
import logging
import logging.config
logging.config.fileConfig('logging.conf')
# create logger
logger = logging.getLogger('simpleExample')
cache = None

def byte2int(bstr, width=32):
  """
  Convert a byte string into a signed integer value of specified width.
  """
  val = sum(ord(b) << 8*n for (n, b) in enumerate(reversed(bstr)))
  if val >= (1 << (width - 1)):
    val = val - (1 << width)
  return val


class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=True, ids=[],
               dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      if labels is not None:
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                  labels.shape))
      else:
          labels = numpy.zeros((images.shape[0],))

      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._ids = ids

  @property
  def ids(self):
    return self._ids

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_file(fileid_path):
  a=[]
  with open(fileid_path, 'rb') as f:
      c = f.read()
      for i in xrange(len(c)):
          cc = byte2int(c[i])
          a.append(cc)
          if cc < 128:
            cc = 0
  if len(a) != 784:
      logger.info("fail to parse image")
      print "xxxxxxxxx faile to parse image", fileid_path
  b = numpy.array(a)
  return b.reshape(28, 28, 1)

def extract_images(path):
    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir(path) if isfile(join(path, f))]
    result = []
    for fileid in files:
      fileid_path = path + '/' + fileid
      b = read_file(fileid_path)
      result.append(b)
    result = numpy.array(result)
    return result, files

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(filename, files, one_hot=True):
    a = {}
    labels = []
    with open(filename, 'r') as f:
      c = csv.reader(f, delimiter= ',')
      for row in c:
        a[row[0]] = int(row[1])

    for fileid in files:
      if fileid not in a:
        logger.info("ERROR xxxxxxxxxxxxxxxx")
        return
      labels.append(a[fileid])
    labels = numpy.array(labels)

    if one_hot:
       return dense_to_one_hot(labels)


def load_data(train_dir="data2", cachename="cache.pkl"):
  TRAIN_IMAGES = train_dir + '/train'
  TRAIN_LABELS = train_dir + '/train.csv'
  TEST_IMAGES = train_dir + '/test'

  try:
    global cache
    if cache:
      logger.info("use global cache %s" % cachename)
    else:
      f = open(cachename, 'rb')
      cache = pickle.load(f)
      f.close()
      logger.info("use cache %s" % cachename)

    (test_images, test_ids,
      train_images, train_labels, train_ids,
    ) = cache

  except:
    train_images, train_files = extract_images(TRAIN_IMAGES)
    train_labels = extract_labels(TRAIN_LABELS, train_files, one_hot=True)
    train_ids = train_files

    test_images, test_files = extract_images(TEST_IMAGES)
    test_ids = test_files


    cache = (test_images, test_ids,
            train_images, train_labels, train_ids,
            )

    f = open(cachename, 'wb')
    pickle.dump(cache, f)
    f.close()

  return {
    'test_images':test_images,
    'test_ids': test_ids,
    'train_images': train_images,
    'train_labels': train_labels,
    'train_ids': train_ids,
  }

def read_data_sets(train_dir, one_hot=True, dtype=tf.float32, validation_size=0):

  class DataSets(object):
    pass
  data_sets = DataSets()

  VALIDATION_SIZE = validation_size
  result = load_data(train_dir)

  test_images = result['test_images']
  test_ids =  result['test_ids']

  train_images = result['train_images']
  train_ids =  result['train_ids']
  train_labels = result['train_labels']

  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]
  train_ids = train_ids[VALIDATION_SIZE:]
  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  validation_ids = train_ids[:VALIDATION_SIZE]


  data_sets.train = DataSet(
      train_images, train_labels,
      ids=train_ids, dtype=dtype
  )

  data_sets.validation = DataSet(
      validation_images, validation_labels,
      ids=validation_ids, dtype=dtype)

  data_sets.test = DataSet(
      test_images, None,
    ids=test_ids, dtype=dtype)

  return data_sets

if __name__ == "__main__":
    load_data()

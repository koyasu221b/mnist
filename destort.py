import numpy
import scipy
import matplotlib.pyplot as plt
import time
from matplotlib.pyplot import imshow
from scipy import ndimage
from parser import load_data, byte2int, read_file, DataSet
import tensorflow as tf
import pickle
from scipy.ndimage.interpolation import geometric_transform
import logging.config
logging.config.fileConfig('logging.conf')
# create logger
logger = logging.getLogger('simpleExample')




def down(b, x):
  return numpy.pad(b, ((x,0),(0,0)), mode="constant")[:-x, :]

def up(b, x):
  return numpy.pad(b, ((0,x),(0,0)), mode="constant")[x:, :]

def right(b,x):
  return numpy.pad(b, ((0,0),(x,0)), mode="constant")[:, :-x]

def left(b, x):
  return numpy.pad(b, ((0,0),(0,x)), mode="constant")[:, x:]

def upper_left(b,y,x):
  return numpy.pad(b, ((0,y),(0,x)), mode="constant")[y:, x:]

def upper_right(b,y,x):
  return numpy.pad(b, ((0,y),(x,0)), mode="constant")[y:, :-x]

def lower_right(b,y,x):
  return numpy.pad(b, ((y,0),(x,0)), mode="constant")[:-y, :-x]

def lower_left(b,y,x):
  return numpy.pad(b, ((y,0),(0,x)), mode="constant")[:-y, x:]

def rotate(b, angle):
  r = ndimage.rotate(b,angle,reshape=False)
  return numpy.clip(r,0, 255)

def dectect_boundary(pic_arr,gap=0, expect=28) :
  tt = [i for i in range(len(pic_arr)) if sum(pic_arr[i])==0]
  tt = [tt[i+1]-tt[i] for i in range(len(tt)-1)]
  tt_max = tt.index(max(tt))+1
  b_t, b_b = len(tt[:tt_max])-gap,len(tt[tt_max:])+1-gap
  tt = [i for i in range(len(pic_arr)) if sum([pic_arr[i][j] for j in range(len(pic_arr))])==0]
  tt = [tt[i+1]-tt[i] for i in range(len(tt)-1)]
  tt_max = tt.index(max(tt))+1
  b_l, b_r = len(tt[:tt_max])-gap,len(tt[tt_max:])+1-gap
  return min([expect,b_t]), min([expect,b_b]), min([expect,b_l]), min([expect,b_r])

def draw_circle(radius):
    grid = numpy.zeros((radius,radius))
    ny, nx = grid.shape
    y, x = numpy.ogrid[:ny, :nx]
    dist = numpy.hypot(x-radius/2, y-radius/2)
    grid[dist <= radius/2] = True
    return (radius,radius),grid

def erosion(b,size,cross):
  return ndimage.grey_erosion(b, size,footprint=cross)

def dilation(b,size,cross):
  return ndimage.grey_dilation(b, size,footprint=cross)
#b = read_file(fileid_path)

def mapping_v1(lc,y,x,dl):
    l,c=lc
    dec=(dl*(l-y))/y
    return l,c+dec

def mapping_v2(lc,y,x,dl):
    l,c=lc
    dec=(dl*(x-l-y))/y
    return l,c+dec

def mapping_h1(lc,y,x,dl):
    c,h=lc
    dec=(dl*(h-x))/x
    return c+dec,h

def mapping_h2(lc,y,x,dl):
    c,h=lc
    dec=(dl*(y-h-x))/x
    return c+dec,h

def shear(a,dl,method,y,x):
    if method=="v1" :
         return geometric_transform(a,mapping_v1,(y,x+dl),order=5,mode='nearest',extra_arguments=(y,x,dl))[:, dl/2:-dl/2]
    elif method=="v2" :
         return geometric_transform(a,mapping_v2,(y,x+dl),order=5,mode='nearest',extra_arguments=(y,x,dl))[:, dl/2:-dl/2]
    elif method=="h1" :
        return geometric_transform(a,mapping_h1,(y+dl,x),order=5,mode='nearest',extra_arguments=(y,x,dl))[dl/2:-dl/2, :]
    elif method=="h2" :
        return geometric_transform(a,mapping_h2,(y+dl,x),order=5,mode='nearest',extra_arguments=(y,x,dl))[dl/2:-dl/2, :]

def noise(a):
  a =  a.reshape(784)
  noise = numpy.random.randint(-64,64 ,size=784)
  a = a + noise
  for i in range(len(a)) :
    if a[i] < 0 :
      a[i]=0
    elif a[i]>255 :
      a[i]=255
  a = a.reshape(28, 28)
  return a

def Crop_UP_DOWN(b,new_up, new_down):
  return numpy.pad( b[new_up:-new_down, : ],((new_up,new_down),(0,0)),mode="constant")

def generate_data(train_dir, use):
  shift = 2
  result = load_data(train_dir)
  train_images = result['train_images']
  train_ids =  result['train_ids']
  train_labels = result['train_labels']
  total = train_images.shape[0]
  #morphology = {"erosion":([1,2],erosion),
  #"dilation":([1,2,3,4],dilation)
  morphology = {"erosion":([2],erosion),
  "dilation":([3],dilation)
  }
  shear_size=20
  generate_images = []
  generate_labels = []
  #actions  = [up, down, left, down, upper_left, upper_right, lower_left, lower_right]
  for i in xrange(total):
    image = train_images[i]
    label = train_labels[i]

    # put original data
    if "default" in use:
      generate_images.append(image)
      generate_labels.append(label)

    if image.shape != (28, 28, 1):
      logger.info("error shape")
      return
    b = image.reshape(28, 28)
    b_T, b_B, b_L, b_R = dectect_boundary(b,0,shift)
    actions  = [(up,[b_T]),
              (down,[b_B]),
              (left,[b_L]),
              (right,[b_R]),]
              #(upper_left,[b_T,b_L]),
              #(upper_right,[b_T,b_R]),
              #(lower_left,[b_B,b_L]),
              #(lower_right,[b_B,b_R])]
    if "shift" in use:
      for action,args in actions:
        t =  action(*tuple([b]+args))
        t = t.reshape(28,28,1)
        generate_images.append(t)
        generate_labels.append(label)

    if "rotate" in use:
      #for angle in [-20,-15,-10,-5,5,10,15,20]:
      for angle in [-15,15]:
        t = rotate(b, angle)
        t = t.reshape(28,28,1)
        generate_images.append(t)
        generate_labels.append(label)

    if "morphology" in use:
      for fps, method in morphology.values():
        for fp in fps :
          size, cross = draw_circle(fp)
          t = method(b,size,cross)
          t = t.reshape(28,28,1)
          generate_images.append(t)
          generate_labels.append(label)

    if "shear" in use:
      y,x=b.shape
      for site in ["v1","v2","h1","h2"]:
        t = shear(b,shear_size,site,y,x)
        t = t.reshape(28,28,1)
        generate_images.append(t)
        generate_labels.append(label)

    if "noise" in use:
      t =  noise(b)
      t = t.reshape(28,28,1)
      generate_images.append(t)
      generate_labels.append(label)

    if "dilation+Crop_UP_DOWN" in use:
      size, cross = draw_circle(3)
      t = dilation(b,size,cross)
      t_T, t_B, t_L, t_R = dectect_boundary(t,0)
      t = Crop_UP_DOWN(t,t_T+3, t_B+3)
      t = t.reshape(28,28,1)
      generate_images.append(t)
      generate_labels.append(label)

  generate_images = numpy.array(generate_images)
  generate_labels = numpy.array(generate_labels)
  return generate_images, generate_labels


def get_generate(train_dir, cachename, use):
  try:
    f = open(cachename, 'rb')
    cache = pickle.load(f)
    (images, labels) = cache
    logger.info("use generate cacheu %s" % cachename)
    f.close()
  except:
    #generate_data(train_dir, use, cachename)
    images , labels = generate_data(train_dir, use)
    cache = (images, labels)
    f = open(cachename, 'wb')
    pickle.dump(cache, f)
    f.close()

  generate = DataSet(
    images, labels,
    ids=[], dtype=tf.float32
  )
  return generate

def get_generate2(train_dir, cachename, use):
  VALIDATION_SIZE = 1000
  try:
    f = open(cachename, 'rb')
    cache = pickle.load(f)
    (images, labels) = cache
    logger.info("use generate cacheu %s" % cachename)
    f.close()
  except:
    images , labels = generate_data(train_dir, use)
    cache = (images, labels)
    f = open(cachename, 'wb')
    pickle.dump(cache, f)
    f.close()

  generate = DataSet(
    images[VALIDATION_SIZE:], labels[VALIDATION_SIZE:],
    ids=[], dtype=tf.float32
  )
  validate = DataSet(
    images[:VALIDATION_SIZE], labels[:VALIDATION_SIZE],
    ids=[], dtype=tf.float32
  )
  return generate, validate

if __name__ == "__main__":
  train_dir = "data2"
  cachename = "generate_cache.pkl"
  get_generate2(train_dir,
                cachename,
                use=[
                  #"shift",
                  #"rotate",
                  #"morphology",
                  "shear",
                  "default"
                ])

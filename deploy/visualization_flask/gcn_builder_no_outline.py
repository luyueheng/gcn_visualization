# GCN model builder for visualization
# lyh

# GCN

import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.activations import sigmoid,softmax
from tensorflow.keras import initializers
from tensorflow.keras.metrics import *
from tensorflow.keras.layers import Input, Embedding, Dense, TimeDistributed, \
                          Dropout,Conv1D, Conv2D, BatchNormalization, \
                          MaxPooling2D,Concatenate,ReLU,LeakyReLU,Reshape,UpSampling3D,\
                          Conv3D,Conv2DTranspose,UpSampling2D
from tensorflow.keras.losses import CosineSimilarity


room2idx = {'Background0': 0, 'Bath0': 1, 'Bath1': 2, 'Bath2': 3, 'Bath3': 4, 
            'Bath4': 5, 'Bath5': 6, 'Bath6': 7, 'Bath7': 8, 'Bath8': 9, 'Bath9': 10, 
            'Bedroom0': 11, 'Bedroom1': 12, 'Bedroom2': 13, 'Bedroom3': 14, 'Bedroom4': 15, 
            'Bedroom5': 16, 'Bedroom6': 17, 'Bedroom7': 18, 'Bedroom8': 19, 'Bedroom9': 20, 
            'Dining0': 21, 'Dining1': 22, 'Dining2': 23, 'Dining3': 24, 'Dining4': 25,
            'Entry0': 26, 'Entry1': 27, 'Entry2': 28, 'Entry3': 29, 'Entry4': 30, 
            'Garage0': 31, 'Garage1': 32, 'Garage2': 33, 'Garage3': 34, 'Garage4': 35,
            'Kitchen0': 36, 'Kitchen1': 37, 'Kitchen2': 38, 'Kitchen3': 39, 'Kitchen4': 40, 
            'LivingRoom0': 41, 'LivingRoom1': 42, 'LivingRoom2': 43, 'LivingRoom3': 44, 'LivingRoom4': 45, 
            'Other0': 46, 'Other1': 47, 'Other10': 48, 'Other11': 49, 'Other12': 50, 
            'Other13': 51, 'Other2': 52, 'Other3': 53, 'Other4': 54, 'Other5': 55, 
            'Other6': 56, 'Other7': 57, 'Other8': 58, 'Other9': 59, 'Outdoor0': 60,
            'Outdoor1': 61, 'Outdoor2': 62, 'Outdoor3': 63, 'Outdoor4': 64, 'Outdoor5': 65, 
            'Outdoor6': 66, 'Outdoor7': 67, 'Outdoor8': 68, 'Outdoor9': 69, 'Storage0': 70, 
            'Storage1': 71, 'Storage2': 72, 'Storage3': 73, 'Storage4': 74, 'Storage5': 75,
            'Storage6': 76, 'Storage7': 77, 'Storage8': 78, 
            'Storage9': 79, 'Entry5':80}


idx2room = dict(zip(list(room2idx.values()),list(room2idx.keys())))


room_color_map = {'LivingRoom':(255,127,0), 
                  'Bedroom':(166,206,227),
                  'Kitchen':(253,191,111),
                  'Dining':(31,120,180),
                  'Bath':(178,223,138),
                  'Storage':(51,160,44),
                  'Entry':(227,26,28),
                  'Garage':(251,154,153),
                  'Other':(202,178,214),
                  'Outdoor':(106,61,154)}



# MLP Layer
def build_mlp(dim_list, activation='relu', batch_norm='batch',
              dropout=0, final_nonlinearity=True):
    ''' Use this function to build mlp networks
        Input: A list of sublayer dimensions
    '''
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(Dense(dim_out,activation="linear",kernel_initializer=initializers.RandomNormal()))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                layers.append(BatchNormalization())
            if activation == 'relu':
                layers.append(ReLU())
            elif activation == 'leakyrelu':
                layers.append(LeakyReLU())
        if dropout > 0:
            layers.append(Dropout(rate=dropout))
    return Sequential(layers)


# GCN Layer
class GraphTripleConv(layers.Layer):
  """
  A single layer of graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='batch',batch_size=32, units = 32):
        

        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.units = units
        
        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
#         self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
#         self.net2.apply(_init_weights)

  def get_config(self):
    return {'units': self.units}

  def call(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (B, O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (B, T, D) giving vectors for all predicates
    - edges: LongTensor of shape (B, T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]
    
    Outputs:
    - new_obj_vecs: FloatTensor of shape (B, O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (B, T, D) giving new vectors for predicates
    """

    O, T = K.int_shape(obj_vecs)[1], K.int_shape(pred_vecs)[1]
    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

    # Break apart indices for subjects and objects; these have shape (B, T,)
    s_idx,o_idx = tf.split(edges,2,axis=2)#shape =(B,T,1)

    s_idx=K.reshape(s_idx,(-1,T)) #shape =(B,T)
    o_idx=K.reshape(o_idx,(-1,T))
    
    i= tf.meshgrid(tf.range(self.batch_size), indexing="ij")
    i= K.reshape(i, (self.batch_size,1))
    i= tf.broadcast_to(i,(self.batch_size,T))
    
    idx_s = tf.stack([i, s_idx], axis=-1)
    idx_o = tf.stack([i, o_idx], axis=-1)
    
    cur_s_vecs = tf.gather_nd(obj_vecs,idx_s)
    cur_o_vecs = tf.gather_nd(obj_vecs,idx_o)
    
    # Get current vectors for triples; shape is (B, T, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (B, T, 2 * H + Dout)
    cur_t_vecs = K.concatenate([cur_s_vecs, pred_vecs, cur_o_vecs], axis=2)
    new_t_vecs = self.net1(cur_t_vecs)
    
    # Break apart into new s, p, and o vecs; s and o vecs have shape (B, T, H) and
    # p vecs have shape (B, T, Dout)
    new_s_vecs = new_t_vecs[:,:, :H]
    new_p_vecs = new_t_vecs[:,:,H:(H+Dout)]
    new_o_vecs = new_t_vecs[:,:,(H+Dout):(2 * H + Dout)]

    # Allocate space for pooled object vectors of shape (B, O, H)
    pooled_obj_vecs =tf.zeros(shape=(self.batch_size,O,H))
    shape=K.shape(pooled_obj_vecs)
    
    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (B, T, H) 
    
    s_idx=K.reshape(s_idx,(-1,T))
    o_idx=K.reshape(o_idx,(-1,T))
       
    i= tf.meshgrid(tf.range(self.batch_size), indexing="ij")
    i= K.reshape(i, (self.batch_size,1))
    i= tf.broadcast_to(i,(self.batch_size,T))

    idx_s = tf.stack([i,s_idx], axis=-1)
    idx_o = tf.stack([i,o_idx], axis=-1)

    pooled_obj_vecs = tf.scatter_nd(idx_s,new_s_vecs,shape=shape)
    pooled_obj_vecs = tf.scatter_nd(idx_o,new_o_vecs,shape=shape) # shape(B, O, H)
        
    if self.pooling == 'avg':
        # Figure out how many times each object has appeared, again using
        # some scatter_add trickery.
        obj_counts = tf.zeros(shape=(self.batch_size,O,H))
        ones = tf.ones(shape=(self.batch_size,T,H))
        
        obj_counts = tf.scatter_nd(idx_s,ones,shape=shape)
        obj_counts = tf.scatter_nd(idx_o,ones,shape=shape)
  
        # Divide the new object vectors by the number of times they
        # appeared, but first clamp at 1 to avoid dividing by zero;
        # objects that appear in no triples will have output vector 0
        # so this will not affect them.
        obj_counts = K.clip(obj_counts,min_value=1,max_value=None)
        pooled_obj_vecs = pooled_obj_vecs / obj_counts

    # Send pooled object vectors through net2 to get output object vectors,
    # of shape (O, Dout)
        new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs


# GCN Net
class GraphTripleConvNet(layers.Layer):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
                   mlp_normalization='batch',name="GCNs",batch_size=32, units = 32):
        super(GraphTripleConvNet, self).__init__()

        self.units = units
        self.num_layers = num_layers
        self.gconvs = []
        gconv_kwargs = {
          'input_dim': input_dim,
          'hidden_dim': hidden_dim,
          'pooling': pooling,
          'mlp_normalization': mlp_normalization,
          'batch_size':batch_size
        }
        for _ in range(self.num_layers):
            self.gconvs.append(GraphTripleConv(**gconv_kwargs))
    
    def call(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs
    
    def get_config(self):
        return {'units': self.units}


# Box Regression
class box_net(layers.Layer):

    def __init__(self, gconv_dim, gconv_hidden_dim=512, box_net_dim=4,mlp_normalization='batch'):
        super(box_net, self).__init__()
        
        self.units = 32
 
        self.box_net_dim = box_net_dim
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)
 
    def call(self, obj_vecs):
        boxes_pred = self.box_net(obj_vecs)
        return boxes_pred
    
    def get_config(self):
        return {'units': self.units}


# Mask Regression
class Mask_regression(layers.Layer):
    """ Mask Regression Layer  """
    def __init__(self, num_objs=35, num_chan=128, mask_size = 64,name="Mask_Regression"):
        super(Mask_regression, self).__init__()
        self.num_objs = num_objs
        self.output_dim = 1
        self.layers, cur_size = [], 1
        self.layers.append(Input(shape=(num_objs, num_chan)))
        self.layers.append(Reshape((num_objs, 1, 1, num_chan)))
        self.units = 32
        while cur_size < mask_size:
            self.layers.append(UpSampling3D(size=(1,2,2)))
            self.layers.append(BatchNormalization())
            self.layers.append(Conv3D(num_chan, kernel_size=(1,3,3), padding='same', activation='relu'))
            cur_size *= 2
        if cur_size != mask_size:
            raise ValueError('Mask size must be a power of 2')
        self.layers.append(Conv3D(self.output_dim, kernel_size=(1,1,1),activation="sigmoid"))
        self.layers.append(Reshape((num_objs, mask_size, mask_size)))
        self.model = Sequential(self.layers)
    
    def call(self, obj_vecs):
        obj_mask= self.model(obj_vecs)
        return obj_mask
    
    def get_config(self):
        return {'units': self.units}

# Rel_Aux_Net
class rel_aux_net(layers.Layer):
    def __init__(self, gconv_out, gconv_hidden_dim, out_dim,
               pooling='avg', mlp_normalization='batch',batch_size=1): 
        super(rel_aux_net, self).__init__()
        self.gconv_out = gconv_out
        self.gconv_hidden_dim = gconv_hidden_dim
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.units = 32

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        self.net_layers = [2 * gconv_out+8, gconv_hidden_dim, out_dim]
        self.net=build_mlp(self.net_layers, batch_norm=mlp_normalization)


    def call(self, obj_vecs_origin, box_pred, edges):
        T = K.int_shape(edges)[1]
        s_idx,o_idx = tf.split(edges,2,axis=-1)#shape =(B,T,1)

        s_idx=tf.squeeze(s_idx,axis=-1) #shape =(B,T)
        o_idx=tf.squeeze(o_idx,axis=-1) 

        i= tf.meshgrid(tf.range(self.batch_size), indexing="ij")
        i= K.reshape(i, (self.batch_size,1))
        i= tf.broadcast_to(i,(self.batch_size,T))

        idx_s = tf.stack([i, s_idx], axis=-1)
        idx_o = tf.stack([i, o_idx], axis=-1)
        
        s_boxes, o_boxes = tf.gather_nd(box_pred,idx_s), tf.gather_nd(box_pred,idx_o)
        s_vecs, o_vecs = tf.gather_nd(obj_vecs_origin,idx_s), tf.gather_nd(obj_vecs_origin,idx_o)
        rel_aux_input = K.concatenate([s_boxes, o_boxes, s_vecs, o_vecs])
        rel_scores = self.net(rel_aux_input)
        rel_scores=softmax(rel_scores, axis=-1)
        rel_scores=K.max(rel_scores, axis=-1, keepdims=False)
        
        return rel_scores
    
    def get_config(self):
        return {'units': self.units}


# Loss Function
def total_loss(boxes_gt, masks_gt, input_p, box_pred, mask_pred, rel_scores,loss):
    y1 = K.flatten(boxes_gt)
    y2= K.flatten(masks_gt)
    y1_pred=K.flatten(box_pred)
    y2_pred=K.flatten(mask_pred)
    input_p=K.expand_dims(input_p,axis=0)
    
    if loss == 'MSE':
        box_loss=losses.MSE(y1, y1_pred)
    else:
        box_loss=losses.MAE(y1, y1_pred)
    mask_loss = losses.BinaryCrossentropy(from_logits=True)(y2, y2_pred)
    cos_sim = losses.CosineSimilarity()(boxes_gt, box_pred)
    
    loss_predicate = losses.categorical_crossentropy(input_p, K.reshape(rel_scores, input_p.shape))
    
    return K.mean(box_loss*10 + 0.01*mask_loss + 0.001*loss_predicate)


def GCN(loss = 'MSE',
        num_objects=80,
        num_relation=3,
        embed_dim=64,
        Din=128,
        H=512,
        Dout=128,
        batch_size=1,
        mask_size = 16,
        num_rooms = 35,
        lr=1e-4,
       ):

    num_edges = int(num_rooms*(num_rooms -1)/2)

    input_o= Input(shape=num_rooms,dtype=tf.int32,batch_size=batch_size)
    input_p=Input(shape=num_edges,dtype=tf.float32,batch_size=batch_size)
    input_t =Input(shape=(num_edges,2),dtype=tf.int32,batch_size=batch_size)

    box_gt=Input(shape=(num_rooms,4),dtype=tf.float32,batch_size=batch_size)
    mask_gt=Input(shape=(num_rooms,mask_size,mask_size),dtype=tf.int32,batch_size=batch_size)

    #Embedding to dense vectors
    embedding_o=Embedding(input_dim=num_objects,output_dim=embed_dim,input_length=num_rooms,mask_zero=True)(input_o)
    embedding_p=Embedding(input_dim=num_relation,output_dim=embed_dim,input_length=num_edges,mask_zero=True)(input_p)

    #Graph Convolutions
    new_s_obj,new_p_obj=GraphTripleConvNet(input_dim=Din, hidden_dim=H,batch_size=batch_size)(embedding_o,embedding_p,input_t)

    #box and mask nets to get scene layout
    output_box=box_net(gconv_dim=Dout)(new_s_obj)
    output_mask = Mask_regression(num_chan=Dout,mask_size = mask_size)(new_s_obj)

    output_rel=rel_aux_net(gconv_out=Dout, gconv_hidden_dim=H,
                           out_dim=num_relation,batch_size=batch_size)(embedding_o,output_box,input_t)


    model = Model([input_o,input_p,input_t,box_gt,mask_gt],[output_box,output_mask,output_rel])

    model.add_loss(total_loss(box_gt, mask_gt ,input_p, output_box, output_mask, output_rel,loss))
    model.compile(optimizer=optimizers.legacy.Adam(learning_rate=lr))
    
    return model


# Helper function for gcn_server
# Prepare json for model prediction
def json_to_data(response):
    '''Response is the graph representation drawn by user'''
    
    data = []
    
    # Rooms
    rooms = response['rooms']
    d = {x: i+1 for i, x in enumerate(rooms)}
    rooms = rooms + [0] * (35 - len(rooms))
    rooms = np.expand_dims(np.array(rooms), axis=0)
    data.append(rooms)
    
    # Weights of relationship
    t = response['triples']
    weights = [x[1] for x in t]
    weights = weights + [0] * (595 - len(weights))
    weights = np.expand_dims(np.array(weights), axis=0)
    data.append(weights)
    
    # Room pairs
    pairs = [[d[x],d[z]] for (x,_,z) in t]
    pairs = pairs + [[0,0]] * (595-len(pairs))
    pairs = np.expand_dims(np.array(pairs), axis=0)
    data.append(pairs)
    
    # Box padding
    data.append(np.expand_dims(np.zeros((35,4), dtype='int'), axis=0))
    # Mask padding
    data.append(np.expand_dims(np.zeros((35,16,16), dtype='int'), axis=0))
    
    return data


# Plot predicted data
def data_to_image(data, pred_data, H=256, W=256, alpha=255):
    
    pred_data = data[:3]+pred_data[:2]

    box = np.squeeze(pred_data[3],0)
    mask = np.squeeze(pred_data[4],0)
    all_rooms = np.squeeze(pred_data[0],0)
    
    room_id = [x for x in all_rooms if x !=0]
    rooms = [idx2room[x] for x in room_id]
    room_types = [room[0:-1] for room in rooms]
    
    colors = [list(room_color_map[r]) for r in room_types]
    
    box_imgs=[]

    for i in range(len(room_id)):
        x0=max(0., box[i][0])
        y0=max(0., box[i][1])
        x1=min(1., box[i][2])
        y1=min(1., box[i][3])
        
        X=np.linspace(int(x0*H),int(x1*H),abs(int(x1*H)-int(x0*H))+1,dtype=int)
        Y=np.linspace(int(y0*W),int(y1*W),abs(int(y1*W)-int(y0*W))+1,dtype=int)
        X=np.sort(X)
        Y=np.sort(Y)
        
        img=np.zeros((H,W,4),dtype=int)

        if (len(X)>1 and len(Y)>1):
            
            dim=(len(Y)-1,len(X)-1)
            mask_rs=cv2.resize(mask[i]*1.,dim)
            mask_rs=np.expand_dims(mask_rs,-1)
            mask_rs=np.broadcast_to(mask_rs,(dim[1],dim[0],4))
            
            img[X[0]:X[-1],Y[0]:Y[-1]]=colors[i]+[alpha]
            img[X[0]:X[-1],Y[0]:Y[-1]]*=mask_rs.astype(int)
            box_imgs.append(img)

    return box_imgs

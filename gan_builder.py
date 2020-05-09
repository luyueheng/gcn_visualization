# GAN model builder for visualization
# lyh

# GAN
import tensorflow as tf
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
               pooling='avg', mlp_normalization='batch',batch_size=32):
        

        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)

  
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


    # Send pooled object vectors through net2 to get output object vectors, of shape (O, Dout)
        new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs


# GCN Net
class GraphTripleConvNet(layers.Layer):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
                   mlp_normalization='batch',name="GCNs",batch_size=32):
        super(GraphTripleConvNet, self).__init__()

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


# Box Regression
class box_net(layers.Layer):

    def __init__(self, gconv_dim, gconv_hidden_dim=512, box_net_dim=4,mlp_normalization='batch'):
        super(box_net, self).__init__()
 
        self.box_net_dim = box_net_dim
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=mlp_normalization)
 
    def call(self, obj_vecs):
        boxes_pred = self.box_net(obj_vecs)
        return boxes_pred


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

# Rel_Aux_Net
class rel_aux_net(layers.Layer):
    def __init__(self, gconv_out, gconv_hidden_dim, out_dim,
               pooling='avg', mlp_normalization='batch',batch_size=1): 
        super(rel_aux_net, self).__init__()
        self.gconv_out = gconv_out
        self.gconv_hidden_dim = gconv_hidden_dim
        self.out_dim = out_dim
        self.batch_size = batch_size

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


# Loss Function
def total_loss(boxes_gt, masks_gt, input_p, box_pred, mask_pred, rel_scores):
    y1 = K.flatten(boxes_gt)
    y2= K.flatten(masks_gt)
    y1_pred=K.flatten(box_pred)
    y2_pred=K.flatten(mask_pred)
    input_p=K.expand_dims(input_p,axis=0)
    
    box_loss=losses.MSE(y1, y1_pred)
    mask_loss = losses.BinaryCrossentropy(from_logits=True)(y2, y2_pred)
    cos_sim = losses.CosineSimilarity()(boxes_gt, box_pred)
    
    loss_predicate = losses.categorical_crossentropy(input_p, rel_scores)
    
    return K.mean(box_loss*1000 + mask_loss + cos_sim + loss_predicate)


# Build Model
# Hyper Parameters - Constant
num_objects = 80
num_relation = 3
embed_dim = 64
Din = 128
H = 512
Dout = 128
mask_size = 16
num_rooms = 35
num_edges = int(num_rooms*(num_rooms-1)/2)
input_size_t = [num_edges,2]



def build_model(batch_size=1, lr=1e-4):
	# Input Layers
	input_o = Input(shape=num_rooms, dtype=tf.int32, batch_size=batch_size)
	input_p = Input(shape=num_edges, dtype=tf.float32, batch_size=batch_size)
	input_t = Input(shape=input_size_t, dtype=tf.int32, batch_size=batch_size)

	box_gt = Input(shape=(num_rooms,4), dtype=tf.float32,batch_size=batch_size)
	mask_gt = Input(shape=(num_rooms,mask_size,mask_size), dtype=tf.int32,batch_size=batch_size)


	# Embeddings
	embedding_o = Embedding(input_dim=num_objects, 
							output_dim=embed_dim, 
							input_length=num_rooms, 
							mask_zero=True)(input_o)
	embedding_p = Embedding(input_dim=num_relation,
							output_dim=embed_dim,
							input_length=num_edges,
							mask_zero=True)(input_p)


	# Graph Convolutions
	new_s_obj, new_p_obj = GraphTripleConvNet(input_dim=Din, hidden_dim=H,
											batch_size=batch_size)(embedding_o,embedding_p,input_t)


	# Box and Mask Regression Nets
	output_box=box_net(gconv_dim=Dout)(new_s_obj)
	output_mask = Mask_regression(num_chan=Dout,mask_size = mask_size)(new_s_obj)

	output_rel=rel_aux_net(gconv_out=Dout, gconv_hidden_dim=H,
	                       out_dim=num_relation,batch_size=batch_size)(embedding_o,output_box,input_t)


	# Model
	model = Model([input_o,input_p,input_t,box_gt,mask_gt], [output_box,output_mask,output_rel])
	model.add_loss(total_loss(box_gt, mask_gt ,input_p, output_box, output_mask, output_rel))
	model.compile(optimizer=optimizers.Adam(learning_rate=lr))

	return model

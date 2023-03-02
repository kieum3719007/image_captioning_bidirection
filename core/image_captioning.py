import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel, RobertaTokenizer
from transformers import ViTFeatureExtractor, TFViTModel
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
import os.path as osp
from core.train import train

def GetRobertaDecoder(pretrained="roberta-base"):
    """
        Get roberta model from Transfomer.
        Param: pretrained - to indicate the pretrained config# [roberta-base, roberta-large]
    """
    configuration = RobertaConfig(is_decoder = True,            #This config to choose the decoder
                                  add_cross_attention = True)   # instead of encoder
    model = TFRobertaModel(configuration)
    model.from_pretrained(pretrained)
    model.layers[0].submodules[1].trainable = False             #This remove the last layer
    return model

def GetRobertaTokenizer(pretrained="roberta-base"):
    """
        Get roberta tokenizer from Transfomer.
        Param: pretrained - to indicate the pretrained config# [roberta-base, roberta-large]
    """
    tokenizer = RobertaTokenizer.from_pretrained(pretrained)
    return tokenizer
    
def GetVitEncoder(pretrained_model='google/vit-base-patch32-224-in21k'):
    """
        Get ViT model from Transfomer.
        Param: pretrained - to indicate the pretrained config
    """
    model = TFViTModel.from_pretrained(pretrained_model)
    model.layers[0].submodules[3].trainable = False  # xem lai tai sao bo 3  lop cuoi
    return model

def GetViTPreprocess(pretrained_model='google/vit-base-patch32-224-in21k'):
    """
        Get ViT preprocessor from Transfomer.
        Param: pretrained - to indicate the pretrained config# [roberta-base, roberta-large]
    """
    model = ViTFeatureExtractor.from_pretrained(pretrained_model)
    return model

class BiTransformerCaptioner(Model):
    
    def __init__(self, config):
        super().__init__()

        self.image_preprocessor = GetViTPreprocess(config["pretrained_model"]["vit"])
        self.image_encoder = GetVitEncoder(config["pretrained_model"]["vit"])

        self.tokenizer = TOKERNIZER
        self.decoder = GetRobertaDecoder(config["pretrained_model"]["roberta"])
        self.bkw_decoder = GetRobertaDecoder(config["pretrained_model"]["roberta"])

        self.token_classifier = Dense(units=self.tokenizer.vocab_size)
        self.token_classifier_bkw = Dense(units=self.tokenizer.vocab_size)
        
    
    def call(self, image, text, text_bkw=None, training=False):        
        encoder_hidden_states = self.image_encoder(**image, training=training).last_hidden_state

        decoder_output = self.decoder(encoder_hidden_states=encoder_hidden_states, **text, training=training)
        output = self.token_classifier(decoder_output.last_hidden_state)

        if (text_bkw):
            decoder_output_bkw = self.bkw_decoder(encoder_hidden_states=encoder_hidden_states, **text_bkw, training=training)
            output_bkw = self.token_classifier_bkw(decoder_output_bkw.last_hidden_state)
            return output, output_bkw
        else:
            return output 
    
def load_weight():
    CKPT = tf.train.Checkpoint(model=MODEL)
    CKPT_MANAGER = tf.train.CheckpointManager(CKPT, CHECKPOINT_PATH, max_to_keep=5)
    CKPT.restore(CKPT_MANAGER.latest_checkpoint)
    print(f'Loaded checkpoint from {CHECKPOINT_PATH}')

PHOBERT_NAME = 'roberta-base'
CHECKPOINT_PATH =osp.join("model", "bi-base-224")


# See all ViT models at https://huggingface.co/models?filter=vit
VIT_MODELS = ["google/vit-base-patch32-384",
              "google/vit-base-patch32-224-in21k",
              "google/vit-base-patch16-224-in21k",
              "google/vit-base-patch16-224",
              "google/vit-base-patch16-384"]

# See all roberta models at https://huggingface.co/models?filter=roberta
ROBERTA_MODELS = ["roberta-base",
                  "roberta-large"]

TOKERNIZER = RobertaTokenizer.from_pretrained(PHOBERT_NAME)
CONFIG = {
    "pretrained_model": {
        "vit": VIT_MODELS[1],
        "roberta": ROBERTA_MODELS[0]
    },
    "tokenizer": TOKERNIZER
}

def scce_with_ls(y, y_hat):
    y = tf.one_hot(tf.cast(y, tf.int32), TOKERNIZER.vocab_size)
    return categorical_crossentropy(y, y_hat, from_logits=True)

loss_object = scce_with_ls

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 1))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

LEARNING_RATE = 2e-5
LOSS = loss_function
OPTIMIZER = tf.keras.optimizers.legacy.Adam(LEARNING_RATE)
MODEL = BiTransformerCaptioner(CONFIG)
MODEL.compile(optimizer='adam', loss=LOSS)
CKPT = tf.train.Checkpoint(model=MODEL,
                           optimizer=OPTIMIZER)

CKPT_MANAGER = tf.train.CheckpointManager(CKPT, CHECKPOINT_PATH, max_to_keep=5)

try:
    load_weight()
    train(MODEL, LOSS, OPTIMIZER, CKPT_MANAGER)
except:
    print("Load weight after train")
    load_weight()

    

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure training/optimization
learning_rate = 0.002   # 学习率
min_word_freq = 3
print_every = 100
chunk_size = 100 # 分组大小
num_labels = 20  # 评价指标个数
num_classes = 4  # number of sentimental types
save_folder = 'models' # 存储的路径

# Configure models 超参数
epochs = 120 # 训练的次数
hidden_size = 500
encoder_n_layers = 2
dropout = 0.05
batch_first = False

# configure path and file name
train_folder = 'data/'
valid_folder = 'data/'
test_a_folder = 'data/'
train_filename = 'trainingset.csv'
valid_filename = 'validationset.csv'
test_a_filename = 'testa.csv'

label_names = ['location_traffic_convenience',
               'location_distance_from_business_district',
               'location_easy_to_find',
               'service_wait_time',
               'service_waiters_attitude',
               'service_parking_convenience',
               'service_serving_speed',
               'price_level',
               'price_cost_effective',
               'price_discount',
               'environment_decoration',
               'environment_noise',
               'environment_space',
               'environment_cleaness',
               'dish_portion',
               'dish_taste',
               'dish_look',
               'dish_recommendation',
               'others_overall_experience',
               'others_willing_to_consume_again']


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3

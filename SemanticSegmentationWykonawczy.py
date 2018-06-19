import sys
from dataset import carlaimgs
from model.erf import net
import tensorflow as tf

#python3 SemanticSegmentationWykonawczy.py 3 5 0.001 256 512 50 ~SCIEZKA_FOLDERU/SemanticSegmantation/out ./model_save/model.ckpt

batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])
learning_rate = float(sys.argv[3])
height = int(sys.argv[4])
width = int(sys.argv[5])
lognumber = int(sys.argv[6])
path = sys.argv[7] #sciezka do folderu z datasetem 'out' w ktorym sa podfoldery z obrazami RGB oraz sciezkami 
output_name = sys.argv[8]


def main(batch_size,epochs,learning_rate,height,width,lognumber,path,output_name):

    def prepare_dataset(dataset, batch_size):
        dataset = (dataset.batch(batch_size)
                   .prefetch(batch_size))
        dataset_iterator = dataset.make_initializable_iterator()
        images, targets = dataset_iterator.get_next()
        return dataset_iterator, images, targets

    image_colors=tf.constant([
        [0,  0, 0],            #0  None
        [139, 131, 134],      #1  Buildings
        [222, 184, 135],    #2  Fences
        [255, 255, 255],      #3  Other
        [255, 193, 37],       #4  Pedestrians
        [173, 255, 47],      #5  Poles
        [255, 255, 0],        #6  RoadLines
        [198,226, 255],      #7  Roads
        [171, 130, 255],      #8  Sidewalks
        [0, 139, 69],         #9  Vegetation
        [255, 0, 0],          #10 Vehicles
        [160, 82, 45],      #11 Walls
        [255, 130, 171],      #12 TrafficSigns
    ],tf.float32)

    train, test, validation = carlaimgs(path,'CameraRGB', 'MyCamera',width=width, height=height)

    #prepare TRAIN
    t_dataset_iterator, t_images, t_labels = prepare_dataset(train,batch_size)

    #prepare TEST
    test_dataset_iterator, test_images, test_labels = prepare_dataset(test,batch_size)

    #prepare VALIDATION
    val_dataset_iterator, val_images, val_labels = prepare_dataset(validation,batch_size)



    #przepuszczenie przez model
    train_output,_ = net(t_images, True, 13, 512, 256)
    val_output,_ = net(val_images, True, 13, 512, 256)
    test_output,_ = net(test_images, True, 13, 512, 256)


    #przygotowanie danych TRAIN
    t_images = tf.transpose(t_images,[0,2,3,1])         #kolor na 4 wymiar
    train_output = tf.transpose(train_output,[0,2,3,1])
    t_labels = tf.transpose(t_labels,[0,2,3,1])

    train_output_softmax=tf.nn.softmax(train_output)    #softmax_wyjscia z sieci
    train_output_res=tf.reshape(train_output,[-1,13])   #reshape na wektor

    train_labels_one_hot = tf.one_hot(t_labels, 13, 1, 0) #one_hot z labels
    train_labels_one_hot=tf.reshape(train_labels_one_hot,[-1,256,512,13]) #pozbycie sie zbednego wymiaru
    train_labels_one_hot_res=tf.reshape(train_labels_one_hot,[-1,13])     #reshape na wektor

    ###########################################################################################################

    #przygotowanie danych VAL
    val_images = tf.transpose(val_images,[0,2,3,1])         #kolor na 4 wymiar
    val_output = tf.transpose(val_output,[0,2,3,1])
    val_labels = tf.transpose(val_labels,[0,2,3,1])

    val_output_softmax=tf.nn.softmax(val_output)    #softmax_wyjscia z sieci
    val_output_res=tf.reshape(val_output,[-1,13])   #reshape na wektor

    val_labels_one_hot = tf.one_hot(val_labels, 13, 1, 0) #one_hot z labels
    val_labels_one_hot=tf.reshape(val_labels_one_hot,[-1,256,512,13]) #pozbycie sie zbednego wymiaru
    val_labels_one_hot_res=tf.reshape(val_labels_one_hot,[-1,13])     #reshape na wektor

    ###########################################################################################################

    test_images = tf.transpose(test_images,[0,2,3,1])         #kolor na 4 wymiar
    test_output = tf.transpose(test_output,[0,2,3,1])
    test_labels = tf.transpose(test_labels,[0,2,3,1])

    test_output_softmax=tf.nn.softmax(test_output)    #softmax_wyjscia z sieci
    test_output_res=tf.reshape(test_output,[-1,13])   #reshape na wektor

    test_labels_one_hot = tf.one_hot(test_labels, 13, 1, 0) #one_hot z labels
    test_labels_one_hot=tf.reshape(test_labels_one_hot,[-1,256,512,13]) #pozbycie sie zbednego wymiaru
    test_labels_one_hot_res=tf.reshape(test_labels_one_hot,[-1,13])     #reshape na wektor

    ###########################################################################################################

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_output_res, labels=train_labels_one_hot_res)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    accu,accu_update_op = tf.metrics.accuracy( labels=tf.argmax(train_labels_one_hot,dimension=3), predictions=tf.argmax(train_output_softmax, dimension=3))
    iou,iou_update_op = tf.metrics.mean_iou( labels=tf.argmax(train_labels_one_hot,dimension=3) , predictions=tf.argmax(train_output_softmax,dimension=3) , num_classes=13)
    # ___________________________________________

    val_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=val_output_res, labels=val_labels_one_hot_res)
    val_loss = tf.reduce_mean(val_cross_entropy)

    val_accu, val_accu_update_op = tf.metrics.accuracy( labels=tf.argmax(val_labels_one_hot,dimension=3), predictions=tf.argmax(val_output_softmax, dimension=3))
    val_iou, val_iou_update_op = tf.metrics.mean_iou( labels=tf.argmax(val_labels_one_hot,dimension=3) , predictions=tf.argmax(val_output_softmax,dimension=3) , num_classes=13)
    # ___________________________________________

    test_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=test_output_res, labels=test_labels_one_hot_res)
    test_loss = tf.reduce_mean(test_cross_entropy)
    test_accu, test_accu_update_op = tf.metrics.accuracy( labels=tf.argmax(test_labels_one_hot,dimension=3), predictions=tf.argmax(test_output_softmax, dimension=3))
    test_iou, test_iou_update_op = tf.metrics.mean_iou( labels=tf.argmax(test_labels_one_hot,dimension=3) , predictions=tf.argmax(test_output_softmax,dimension=3) , num_classes=13)
 
    #  ___________________________________________
    # tensorboard
    tb_train_loss = tf.summary.scalar('metrics/loss', loss)
    tb_train_accu = tf.summary.scalar('metrics/accu', accu)
    tb_train_iou = tf.summary.scalar('metrics/iou', iou)

    train_output_argmax=tf.argmax(train_output,dimension=3)
    train_output_argmax=tf.reshape(train_output_argmax,[-1,256,512])

    t_labels=tf.cast(tf.squeeze(t_labels,3),tf.int32)
    image_labels=tf.nn.embedding_lookup(image_colors,t_labels)
    imageout=tf.nn.embedding_lookup(image_colors,train_output_argmax) #pokolorowanie obrazkow

    before_train_imgs = tf.summary.image('metrics/before_train_image', t_images)   #zapis do tensorboard
    after_train_imgs = tf.summary.image('metrics/after_train_image', imageout)
    labels = tf.summary.image('metrics/labels', image_labels)

    stats_train = tf.summary.merge([before_train_imgs, after_train_imgs,labels])
    stats_train_loss = tf.summary.merge([tb_train_loss,tb_train_accu,tb_train_iou])

    fwtrain = tf.summary.FileWriter(logdir='./training', graph=tf.get_default_graph())

    #    ___________________________________________

    tb_val_loss = tf.summary.scalar('metrics/val_loss', val_loss)
    tb_val_accu = tf.summary.scalar('metrics/val_accu', val_accu)
    tb_val_iou = tf.summary.scalar('metrics/val_iou', val_iou)

    val_output_argmax=tf.argmax(val_output,dimension=3)
    val_output_argmax=tf.reshape(val_output_argmax,[-1,256,512])

    val_labels=tf.cast(tf.squeeze(val_labels,3),tf.int32)
    val_image_labels=tf.nn.embedding_lookup(image_colors,val_labels)
    val_imageout=tf.nn.embedding_lookup(image_colors,val_output_argmax) #pokolorowanie obrazkow

    val_after_train_imgs = tf.summary.image('metrics/val_after_train_image', val_imageout)
    before_vali_imgs = tf.summary.image('metrics/before_vali_image', val_images)     
    vali_labels = tf.summary.image('metrics/vali_labels', val_image_labels)

    val_stats_train = tf.summary.merge([tb_val_loss, tb_val_accu, tb_val_iou, before_vali_imgs, val_after_train_imgs, vali_labels])
    val_fwtrain = tf.summary.FileWriter(logdir='./validation', graph=tf.get_default_graph())

    #    ___________________________________________

    tb_test_loss = tf.summary.scalar('metrics/test_loss', test_loss)    
    tb_test_accu = tf.summary.scalar('metrics/test_accu', test_accu)
    tb_test_iou = tf.summary.scalar('metrics/test_iou', test_iou)

    test_output_argmax=tf.argmax(test_output,dimension=3)
    test_output_argmax=tf.reshape(test_output_argmax,[-1,256,512])

    test_labels=tf.cast(tf.squeeze(test_labels,3),tf.int32)
    test_image_labels=tf.nn.embedding_lookup(image_colors,test_labels)
    test_imageout=tf.nn.embedding_lookup(image_colors,test_output_argmax) #pokolorowanie obrazkow

    test__imgs = tf.summary.image('metrics/test__imgs', test_images)  
    test_after_train_imgs = tf.summary.image('metrics/test_after_train_image', test_imageout)    
    test_img_labels = tf.summary.image('metrics/test_img_labels', test_image_labels)
    test_stats_train = tf.summary.merge([tb_test_loss, tb_test_accu, tb_test_iou, test__imgs, test_after_train_imgs, test_img_labels])

    test_fwtrain = tf.summary.FileWriter(logdir='./testing', graph=tf.get_default_graph())
    
    saver=tf.train.Saver()
    iou_max=0 #zmienna przechowujaca najwieksze iou dla epok
    epoch_sum_value=0 #suma iou w epoce

    #    ___________________________________________
    with tf.Session() as sess:
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_g)
            sess.run(init_l)
            #running
            i = 0
            j = 0
            k = 0

            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
            running_vars_initializer = tf.variables_initializer(var_list=running_vars)

            for epoch in range(epochs):

                sess.run([running_vars_initializer])
                counter_iou=0
                epoch_sum_value=0

                sess.run(t_dataset_iterator.initializer)
                while True:
                    try:
                        _,x_stats_train = sess.run([optimizer , stats_train_loss])


                        if i%lognumber == 0:
                            _,_=sess.run([accu_update_op,iou_update_op])
                            current_accu,current_iou = sess.run([accu,iou])

                            fwtrain.add_summary(x_stats_train, i)
                            print('krok:\t',i,'\t || \t iou:\t',current_iou,'\t accu:\t',current_accu)

                            stats_train_b=sess.run(stats_train)
                            fwtrain.add_summary(stats_train_b, i)

                        i += 1

                    except tf.errors.OutOfRangeError:
                        break


                sess.run(val_dataset_iterator.initializer)
                while True:
                    try:

                        val_stats_train_x = sess.run(val_stats_train)
                        
                        if j%20 == 0:

                            _,_=sess.run([val_accu_update_op, val_iou_update_op])
                            val_current_accu,val_current_iou = sess.run([val_accu, val_iou])

                            val_fwtrain.add_summary(val_stats_train_x, j)

                            epoch_sum_value += val_current_iou

                            counter_iou+=1
                            print('krok:\t',j,'\t || \t val_iou:\t',val_current_iou,'\t val_accu:\t',val_current_accu)
                            
                        j += 1
                    except tf.errors.OutOfRangeError:
                        break

                epoch_iou_value=epoch_sum_value/counter_iou
                print('---->>\tSrednie IOU dla epoki:\t',epoch,"\twynioslo:\t",epoch_iou_value,'\t<<-----')
                if(epoch_iou_value > iou_max): #and epoch > 0.7*epochs
                    iou_max=epoch_iou_value
                    saver.save(sess,output_name)
                    print("Zapisano model dla epoki: ",epoch)

                #print("\t|||\tZakonczona epoka:", epoch, '\t|||')


            sess.run(test_dataset_iterator.initializer)
            while True:

                try:
                    sess.run([running_vars_initializer])

                    test_stats_train_x = sess.run(test_stats_train)

                    _,_=sess.run([test_accu_update_op, test_iou_update_op])
                    test_current_accu, test_current_iou = sess.run([test_accu, test_iou])

                    test_fwtrain.add_summary(test_stats_train_x, k)
                    print('krok:\t',k,'\t || \t test_iou:\t',test_current_iou,'\t test_accu:\t',test_current_accu)

                    k += 1
                except tf.errors.OutOfRangeError:
                    break
            print("Zakonczono!")
    return 0

main(batch_size,epochs,learning_rate,height,width,lognumber,path,output_name)
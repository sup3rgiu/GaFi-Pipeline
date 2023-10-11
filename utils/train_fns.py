import tensorflow as tf
from tensorflow.keras import backend as K
from utils import misc, augmentation, optimizers
import utils.callbacks
from models import classifiers
import utils.config
import utils.callbacks
import os
import gc
import numpy as np
import pandas as pd
import json
import glob
import shutil
from PIL import Image
import matplotlib.pyplot as plt
plt.rc('font', size=16)

tfk = tf.keras

def train_gan(
    gan,
    training_dataset,
    epochs,
    config,
    steps_per_epoch=None,
    run_name='',
    show_plots=False,
    plot_with_title=False,
    plot_title='',
    callbacks=[],
    validation_data=None,
    save=True,
    keep_images=False,
    sample_ema=True,
    apply_standing_stats=False,
    fixed_seed=False,
    reload=False,
    **kwargs
):
    
    print(f"\nTraining model with name: {run_name}\n")
    
    initial_epoch = 0
    base_save_path = os.path.join(config.RUN.save_path, config.DATA.dataset)
    save_path = os.path.join(base_save_path, run_name, 'last_epoch')

    images_save_path = os.path.join('./save/Images', config.DATA.dataset, run_name)
       
    if apply_standing_stats:
        print("During dataset generator, standing statistics trick will be applied (safely, on a copy of the training model)\n")
            
    if sample_ema and not hasattr(gan, 'ema_generator'):
        print("'sample_ema' was True but it has been reversed to False because the current model has no EMA Generator\n")
        sample_ema = False
    
    ### Create callbacks ###
    sample_images_cb = utils.callbacks.SampleImages(config=config, save_path=images_save_path, title=plot_title, image_num=5, plot_with_title=plot_with_title,
                                                    sample_ema=sample_ema, save_plot=save, show_plots=show_plots, apply_standing_stats=apply_standing_stats)
    callbacks.append(utils.callbacks.TerminateOnNaN(contains=''))
    hist_df_cb = utils.callbacks.HistoryDataframe()
    callbacks.append(hist_df_cb)
    callbacks.append(tfk.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=['lr', 'lr_g', 'lr_d']))
    callbacks.append(sample_images_cb)
    if fixed_seed:
        # Fix the same seed at the beginning of each epoch
        # In this way, the "random" images generated during model training are always the same (between epochs, not between steps/batches)
        callbacks.append(utils.callbacks.FixSeed(seed=config.RUN.seed))
    #callbacks_list = tfk.callbacks.CallbackList(callbacks=callbacks, add_history=False, add_progbar=False, model=gan)

    ### Reload model if needed ###
    # Callbacks properties are relaoded too
    if reload:
        print (f"Reloading model '{run_name}'")
        reload_dict = misc.reload_gan_model(gan=gan, model_save_path=save_path, backup_files=True, callbacks=callbacks,)
        gan = reload_dict['gan']
        initial_epoch = reload_dict['epoch']
    
    ### Train the model ###
    try:
        hist = gan.fit(training_dataset, validation_data=validation_data, epochs=epochs, initial_epoch=initial_epoch,
                       steps_per_epoch=steps_per_epoch, verbose=1, callbacks=callbacks, **kwargs).history          
    except KeyboardInterrupt:
        print('\n\n *** Interrupting training *** \n\n')

    gan_history = hist_df_cb.history
    
    if save:
        print(f"Saving model and stuff...")
        ### Save models and history ###
        os.makedirs(save_path, exist_ok=True)
        gan_history.to_csv(save_path+'/history.csv', index=False)
        gan.generator.save(save_path+'/generator', include_optimizer=config.OPTIMIZATION.save_optimizer)
        gan.discriminator.save(save_path+'/discriminator', include_optimizer=config.OPTIMIZATION.save_optimizer)
        if hasattr(gan, 'ema_generator'):
            gan.ema_generator.save(save_path+'/ema_generator', include_optimizer=config.OPTIMIZATION.save_optimizer) # actually ema_generator does not have an optimizer
        with open(save_path+'/config.json', 'w') as fp:
            json.dump(config, fp=fp, cls=utils.config.ConfigEncoder, indent=4)

        ### Convert images to a single gif ###
        # We could delegate this to the SampleImages callback in the on_train_end function,
        # but I prefer to do it here to save the important things (i.e. models) first
        fp_in = sample_images_cb.save_path+"/*.png"
        extra_number = len(glob.glob(os.path.join(save_path, 'images*.gif'))) + 1 # needed to not overwrite the already existing gif when 'reload' is True and 'backup_files' in misc.reload_model() is False
        fp_out = os.path.join(save_path, f"images{extra_number if extra_number != 1 else ''}.gif")
        imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
        img = next(imgs)
        img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=100, optimize=False)
        if not keep_images:
            shutil.rmtree(sample_images_cb.save_path)

    ### Plot metrics history ###
    if show_plots:
        plt.rc('font', size=20) 
        for col in gan_history.columns:
            if(col[:3]!='val'):
                plt.figure(figsize=(24,4))
                plt.plot(gan_history[col], label=col, alpha=.8, color='#FFA522')
                if 'val_'+col in gan_history.columns:
                    plt.plot(gan_history['val_'+col], label='val_'+col, alpha=.8, color='#4081EA')
                plt.legend()
                plt.grid(alpha=.3)
        plt.show()
    
    return {
        "generator": gan.generator,
        "discriminator": gan.discriminator,
        "ema_generator": gan.ema_generator if hasattr(gan, 'ema_generator') else None,
        "model_history": gan_history,
        "callbacks": callbacks
    }

def run_pipeline_step(
        generator,
        class_samples,
        testing_dataset,
        classifier_save_path,
        cfgs,
        classifier_pretrained=None,
        filter=False,
        threshold=0.0,
        stddev=1.0,
        fixed_dset=True,
        recycling_period=10,
        recycling_factor=1,
    ):

    assert not filter or classifier_pretrained is not None, "If filter is True, then classifier_pretrained must not be None"
    
    gen = generator
    batch_size = cfgs.OPTIMIZATION.batch_size
    epochs = cfgs.OPTIMIZATION.epochs

    ### Build optimizer ###
    optimizer = optimizers.get_optimizer(optimizer=cfgs.OPTIMIZATION.optimizer, lr=cfgs.OPTIMIZATION.lr, momentum=cfgs.OPTIMIZATION.momentum,
                                         nesterov=cfgs.OPTIMIZATION.nesterov, weight_decay=cfgs.OPTIMIZATION.weight_decay)

    ### Build and compile classifier ###
    criterion = tf.keras.losses.CategoricalCrossentropy()
    input_shape = (cfgs.DATA.image_size, cfgs.DATA.image_size, cfgs.DATA.num_channels)
    augmentation_layer = augmentation.augment(image_size=cfgs.DATA.image_size, normalize=cfgs.DATA.normalize, random_flip=cfgs.AUG.random_flip, random_crop=cfgs.AUG.random_crop,
                                              random_rotation=cfgs.AUG.random_rotation, random_zoom=cfgs.AUG.random_zoom, random_erasing=cfgs.AUG.random_erasing, autocast=False)
    classifier_fake = classifiers.get_classifier(model=cfgs.MODEL.name, input_shape=input_shape, num_classes=cfgs.DATA.num_classes, width=cfgs.MODEL.width,
                                                 seed=cfgs.RUN.seed, augmentation_layer=augmentation_layer)
    classifier_fake.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])

    ### Build full run name ###
    config_details = f"""{f'DS{recycling_period}_r{recycling_factor}' if not fixed_dset else '_fixed'}_std{stddev}{f'_filter{threshold}' if filter else ''}"""
    classifier_full_save_path = classifier_save_path + '_' + config_details
    ckpt_name = os.path.basename(classifier_full_save_path)

    print(f"c_path: {classifier_full_save_path}\n")

    misc.print_gpu_usage()

    ### Generate initial fake dataset ###
    fake_dataset, fake_labels = misc.generate_dataset(model=gen, num_classes=cfgs.DATA.num_classes, filter=filter,
                                                      class_samples=class_samples, batch_size=batch_size,
                                                      classifier_pretrained=classifier_pretrained,
                                                      threshold=threshold, stddev=stddev, filtering_attempts=cfgs.PIPELINE.filtering_attempts,
                                                      seed=cfgs.RUN.seed, verbose=True)
        
    ### Build callbacks ###
    callbacks = [
        tfk.callbacks.ModelCheckpoint(filepath=classifier_full_save_path, save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True),
        utils.callbacks.LearningRateEpochScheduler(utils.callbacks.lr_scheduler(milestones=[60, 80], gamma=0.1)),
        #tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', restore_best_weights=True, patience=10),
        tfk.callbacks.ProgbarLogger(count_mode='steps'), # manually add ProgbarLogger to decide its position w.r.t. other callbacks (otherwise it's always the last callback and so other callbacks' prints could interrupt the progbar during validation step)
        utils.callbacks.LogBestAccuracyCallback(),
    ]
    
    ### Run training loop ###
    history_df = pd.DataFrame()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        misc.print_gpu_usage()

        # train classifier on fake dataset for one epoch
        _history = classifier_fake.fit(
                    x=fake_dataset,
                    y=fake_labels,
                    epochs=1,
                    validation_data=(testing_dataset),
                    shuffle=True,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=callbacks
                ).history

        # Update global history
        if(history_df.empty):
            history_df = pd.DataFrame(columns=list(_history.keys()))
        history_df.loc[len(history_df.index)] = np.squeeze(list(_history.values()))
        
        # Avoid new dataset generation at last epoch
        if epoch == epochs-1:
            break

        if not fixed_dset and ((epoch+1) % recycling_period == 0):

            # Remove part of the previous fake dataset
            sub_fake_dataset, sub_fake_labels = misc.subsample_dataset(fake_dataset, fake_labels, samples_per_class=int((1-recycling_factor)*class_samples))
            del fake_dataset, fake_labels # free up space?
            K.clear_session()
            gc.collect()

            # Generate new (possibly smaller) fake dataset
            new_dataset_samples = int(recycling_factor * class_samples)
            new_fake_dataset, new_fake_labels = misc.generate_dataset(model=gen, num_classes=cfgs.DATA.num_classes, filter=filter,
                                                                      class_samples=new_dataset_samples, batch_size=batch_size,
                                                                      classifier_pretrained=classifier_pretrained,
                                                                      threshold=threshold, stddev=stddev, filtering_attempts=cfgs.PIPELINE.filtering_attempts,
                                                                      verbose=True, seed=None) # seed=None to use a new random seed

            if recycling_factor == 1:
                fake_dataset = new_fake_dataset
                fake_labels = new_fake_labels
            else:
                # Merge the new dataset and the subsampled dataset (the shuffle will be done inside .fit())
                fake_dataset = np.vstack([sub_fake_dataset, new_fake_dataset])
                fake_labels = np.vstack([sub_fake_labels, new_fake_labels])

            assert fake_dataset.shape[0] == class_samples * cfgs.DATA.num_classes, "Something wrong"
            del new_fake_dataset, new_fake_labels, sub_fake_dataset, sub_fake_labels # free up space?
            K.clear_session()
            gc.collect() # very important to not keep increasing the GPU memory usage. Sort of memory leak which happens when you do multiple .predict() calls (which happens during fake dataset generation)
     
    del fake_dataset, fake_labels # free up space?
    K.clear_session()
    gc.collect()

    ### Save result ###
    results_path = os.path.join(os.path.dirname(classifier_full_save_path), 'results.csv')
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path, header=0)
    else:
        columns = ['checkpoint_name','val_accuracy']
        results_df = pd.DataFrame(columns=columns)

    results_df.loc[len(results_df.index)] = [ckpt_name, max(history_df['val_accuracy'])]
    results_df.to_csv(results_path, index=False)

    ### Save history ###
    history_path = os.path.join(os.path.dirname(classifier_full_save_path), 'histories.csv')
    if os.path.exists(history_path):
        histories_df = pd.read_csv(history_path, header=0)
    else:
        columns = ['checkpoint_name','history']
        histories_df = pd.DataFrame(columns=columns)

    history_df.index.name = 'epoch'
    history_df.index = history_df.index + 1 # start from 1 as the epochs

    histories_df.loc[len(histories_df.index)] = [ckpt_name, history_df.to_csv(sep=' ').strip('\n')] # .strip('\n') to remove last \n
    histories_df.to_csv(history_path, index=False)

    # We can later reload the history as a DataFrame with:
    #
    # loaded_df = pd.read_csv(history_path)
    # data = loaded_df['history'][0] # select the first row (as example)
    # loaded_history = pd.DataFrame([x.split(' ') for x in data.split('\n')[1:]], columns=[x for x in data.split('\n')[0].split(' ')]) # convert csv string to DataFrame
    # loaded_history.set_index('epoch', inplace=True)

    return {
        'classifier_fake': classifier_fake,
        'history': history_df
    }

def run_multigan_step(
        gen_configs,
        class_samples,
        testing_dataset,
        classifier_save_path,
        cfgs,
        classifier_pretrained=None,
        filter=False,
        fixed_dset=True,
        recycling_period=1,
        recycling_factor=1,
    ):

    assert not filter or classifier_pretrained is not None, "If filter is True, then classifier_pretrained must not be None"
    
    batch_size = cfgs.OPTIMIZATION.batch_size
    epochs = cfgs.OPTIMIZATION.epochs
    num_gen = len(gen_configs)
    one_gan_for_epoch = cfgs.PIPELINE.one_gan_for_epoch

    if one_gan_for_epoch:
        class_samples_per_gen = class_samples
    else:
        class_samples_per_gen = class_samples // num_gen

    ### Build optimizer ###
    optimizer = optimizers.get_optimizer(optimizer=cfgs.OPTIMIZATION.optimizer, lr=cfgs.OPTIMIZATION.lr, momentum=cfgs.OPTIMIZATION.momentum,
                                         nesterov=cfgs.OPTIMIZATION.nesterov, weight_decay=cfgs.OPTIMIZATION.weight_decay)

    ### Build and compile classifier ###
    criterion = tf.keras.losses.CategoricalCrossentropy()
    input_shape = (cfgs.DATA.image_size, cfgs.DATA.image_size, cfgs.DATA.num_channels)
    augmentation_layer = augmentation.augment(image_size=cfgs.DATA.image_size, normalize=cfgs.DATA.normalize, random_flip=cfgs.AUG.random_flip, random_crop=cfgs.AUG.random_crop,
                                              random_rotation=cfgs.AUG.random_rotation, random_zoom=cfgs.AUG.random_zoom, random_erasing=cfgs.AUG.random_erasing, autocast=False)
    classifier_fake = classifiers.get_classifier(model=cfgs.MODEL.name, input_shape=input_shape, num_classes=cfgs.DATA.num_classes, width=cfgs.MODEL.width,
                                                 seed=cfgs.RUN.seed, augmentation_layer=augmentation_layer)
    classifier_fake.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])

    ### Build full run name ###
    run_name = 'seed'
    for k, v in gen_configs.items():
        run_name += '_' + str(v['seed'])
    classifier_full_save_path = os.path.join(classifier_save_path, run_name)
    existing_runs = len(glob.glob(classifier_full_save_path))
    if existing_runs > 0:
        extra_name = f'_run{existing_runs + 1}'
        print(f"{existing_runs} runs with the same name already exist. Saving in a new folder ('{extra_name}')")
        classifier_full_save_path += extra_name
    print(f"c_path: {classifier_full_save_path}\n")

    ### Generate initial fake dataset ###
    if one_gan_for_epoch:
        k, v = list(gen_configs.items())[0]
        print(f"Using {k}: ckpt epoch {v['ckpt_epoch']} with stddev {v['stddev']} and threshold {v['threshold']}")
        fake_dataset, fake_labels = misc.generate_dataset(model=v['gen'], num_classes=cfgs.DATA.num_classes, filter=filter,
                                                          class_samples=class_samples_per_gen, batch_size=batch_size,
                                                          classifier_pretrained=classifier_pretrained,
                                                          threshold=v['threshold'], stddev=v['stddev'], filtering_attempts=cfgs.PIPELINE.filtering_attempts,
                                                          seed=cfgs.RUN.seed, verbose=True)
    else:
        fake_datasets, fake_labels = [], []
        for k, v in gen_configs.items():
            print(f"Using {k}: ckpt epoch {v['ckpt_epoch']} with stddev {v['stddev']} and threshold {v['threshold']}")
            tmp_ds, tmp_lbl = misc.generate_dataset(model=v['gen'], num_classes=cfgs.DATA.num_classes, filter=filter,
                                                    class_samples=class_samples_per_gen, batch_size=batch_size,
                                                    classifier_pretrained=classifier_pretrained,
                                                    threshold=v['threshold'], stddev=v['stddev'], filtering_attempts=cfgs.PIPELINE.filtering_attempts,
                                                    seed=cfgs.RUN.seed, verbose=True)
            fake_datasets.append(tmp_ds)
            fake_labels.append(tmp_lbl)
        fake_dataset = np.vstack(fake_datasets)
        fake_labels = np.vstack(fake_labels)
        
    ### Build callbacks ###
    callbacks = [
        tfk.callbacks.ModelCheckpoint(filepath=classifier_full_save_path, save_weights_only=False, monitor='val_accuracy', mode='max', save_best_only=True),
        utils.callbacks.LearningRateEpochScheduler(utils.callbacks.lr_scheduler(milestones=[60, 80], gamma=0.1)),
        #tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', restore_best_weights=True, patience=10),
        tfk.callbacks.ProgbarLogger(count_mode='steps'), # manually add ProgbarLogger to decide its position w.r.t. other callbacks (otherwise it's always the last callback and so other callbacks' prints could interrupt the progbar during validation step)
        utils.callbacks.LogBestAccuracyCallback(),
    ]
    
    ### Run training loop ###
    history_df = pd.DataFrame()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))

        # train classifier on fake dataset for one epoch
        _history = classifier_fake.fit(
                    x=fake_dataset,
                    y=fake_labels,
                    epochs=1,
                    validation_data=(testing_dataset),
                    shuffle=True,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=callbacks
                ).history

        # Update global history
        if(history_df.empty):
            history_df = pd.DataFrame(columns=list(_history.keys()))
        history_df.loc[len(history_df.index)] = np.squeeze(list(_history.values()))
        
        # Avoid new dataset generation at last epoch
        if epoch == epochs-1:
            break

        if not fixed_dset and ((epoch+1) % recycling_period == 0):

            # Remove part of the previous fake dataset
            sub_fake_dataset, sub_fake_labels = misc.subsample_dataset(fake_dataset, fake_labels, samples_per_class=int((1-recycling_factor)*class_samples_per_gen*num_gen))
            del fake_dataset, fake_labels # free up space?
            K.clear_session()
            gc.collect()

            # Generate new (possibly smaller) fake dataset
            new_dataset_samples = int(recycling_factor * class_samples_per_gen)

            if one_gan_for_epoch:
                index = (epoch+1) % num_gen
                k, v = list(gen_configs.items())[index]
                print(f"Using {k}: ckpt epoch {v['ckpt_epoch']} with stddev {v['stddev']} and threshold {v['threshold']}")
                fake_dataset, fake_labels = misc.generate_dataset(model=v['gen'], num_classes=cfgs.DATA.num_classes, filter=filter,
                                                                  class_samples=new_dataset_samples, batch_size=batch_size,
                                                                  classifier_pretrained=classifier_pretrained,
                                                                  threshold=v['threshold'], stddev=v['stddev'], filtering_attempts=cfgs.PIPELINE.filtering_attempts,
                                                                  verbose=True, seed=None) # seed=None to use a new random seed
            else:
                fake_datasets, fake_labels = [], []
                for k, v in gen_configs.items():
                    #print(f"Using {k}: ckpt epoch {v['ckpt_epoch']} with stddev {v['stddev']} and threshold {v['threshold']}")
                    tmp_ds, tmp_lbl = misc.generate_dataset(model=v['gen'], num_classes=cfgs.DATA.num_classes, filter=filter,
                                                            class_samples=new_dataset_samples, batch_size=batch_size,
                                                            classifier_pretrained=classifier_pretrained,
                                                            threshold=v['threshold'], stddev=v['stddev'], filtering_attempts=cfgs.PIPELINE.filtering_attempts,
                                                            verbose=True, seed=None) # seed=None to use a new random seed
                    fake_datasets.append(tmp_ds)
                    fake_labels.append(tmp_lbl)
                new_fake_dataset = np.vstack(fake_datasets)
                new_fake_labels = np.vstack(fake_labels)
                del fake_datasets, fake_labels

            if recycling_factor == 1:
                fake_dataset = new_fake_dataset
                fake_labels = new_fake_labels
            else:
                # Merge the new dataset and the subsampled dataset (the shuffle will be done inside .fit())
                fake_dataset = np.vstack([sub_fake_dataset, new_fake_dataset])
                fake_labels = np.vstack([sub_fake_labels, new_fake_labels])

            if one_gan_for_epoch:
                assert fake_dataset.shape[0] == class_samples_per_gen * cfgs.DATA.num_classes, "Something wrong"
            else:
                assert fake_dataset.shape[0] == class_samples_per_gen * cfgs.DATA.num_classes * num_gen, "Something wrong"

            del new_fake_dataset, new_fake_labels, sub_fake_dataset, sub_fake_labels # free up space?
            K.clear_session()
            gc.collect() # very important to not keep increasing the GPU memory usage. Sort of memory leak which happens when you do multiple .predict() calls (which happens during fake dataset generation)
     
    del fake_dataset, fake_labels # free up space?
    K.clear_session()
    gc.collect()

    ### Save result ###
    results_path = os.path.join(classifier_full_save_path, 'results.csv')
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path, header=0)
    else:
        columns = ['val_accuracy']
        results_df = pd.DataFrame(columns=columns)

    results_df.loc[len(results_df.index)] = [max(history_df['val_accuracy'])]
    results_df.to_csv(results_path, index=False)

    ### Save history ###
    history_path = os.path.join(classifier_full_save_path, 'histories.csv')
    if os.path.exists(history_path):
        histories_df = pd.read_csv(history_path, header=0)
    else:
        columns = ['history']
        histories_df = pd.DataFrame(columns=columns)

    history_df.index.name = 'epoch'
    history_df.index = history_df.index + 1 # start from 1 as the epochs

    histories_df.loc[len(histories_df.index)] = [history_df.to_csv(sep=' ').strip('\n')] # .strip('\n') to remove last \n
    histories_df.to_csv(history_path, index=False)

    # We can later reload the history as a DataFrame with:
    #
    # loaded_df = pd.read_csv(history_path)
    # data = loaded_df['history'][0] # select the first row (as example)
    # loaded_history = pd.DataFrame([x.split(' ') for x in data.split('\n')[1:]], columns=[x for x in data.split('\n')[0].split(' ')]) # convert csv string to DataFrame
    # loaded_history.set_index('epoch', inplace=True)

    ### Save run config ###
    run_config_file_path = os.path.join(classifier_full_save_path, 'run_config.json')
    with open(run_config_file_path, "w") as write_file:
        json.dump(gen_configs, write_file, default=lambda o: '<not serializable>', indent=4)

    config_file_path = os.path.join(classifier_full_save_path, 'config.json')
    with open(config_file_path, 'w') as fp:
        json.dump(cfgs, fp=fp, cls=utils.config.ConfigEncoder, indent=4)

    return {
        'classifier_fake': classifier_fake,
        'history': history_df
    }
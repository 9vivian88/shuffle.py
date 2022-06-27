from torch.utils.data import DataLoader
import dataset.train_dataset
import dataset.test_dataset


def kon10k_1000(config):
    print("kon10k 1000 bullied successfully   " + str(config.split))
    if config.data_mode == 'train':
        # train_data = dataset.train_dataset.train_data('./data/kon10k/train1000/labeled/labeled_' + str(config.split),/home/ubuntu/nashome/data/my retouch/retouch_0508/label1-2000_16/mosv1
        #                                               './data/kon10k/1024x768')/home/ubuntu/nashome/data/my retouch/retouch_0508/label1-2000_16/labelMOS
        train_data = dataset.train_dataset.train_data('/home/ubuntu/nashome/data/my retouch/retouch_0508/label1-2000_16/mosv2/n1label_2_1-2000_train.txt',
                                                      '/home/ubuntu/nashome/data/my retouch/retouch_0508/hdr1-2000')

        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))
        return train_dataloader

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('/home/ubuntu/nashome/data/my retouch/retouch_0508/label1-2000_16/mosv2/n1label_2_1-2000_test.txt',
                                                   '/home/ubuntu/nashome/data/my retouch/retouch_0508/hdr1-2000')
        test_dataloader = DataLoader(test_data,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(config.batch_size * len(test_dataloader)))
        return test_dataloader

    if config.data_mode == 'semi':
        # train_data = dataset.train_dataset.train_data(
        #     './data/kon10k/train1000/labeled/labeled_' + str(config.split),
        #     './data/kon10k/1024x768')
        train_data = dataset.train_dataset.train_data(
            './data/mei1.6/train600/label/train',
            './data/mei1.6/train600/512x512')
        unlabel_dataset = dataset.train_dataset.train_data(
            './data/kon10k/train1000/unlabeled/unlabeled_' + str(config.split),
            './data/kon10k/1024x768')
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=False)
        unlabel_data_loader = DataLoader(unlabel_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         num_workers=config.number_workers,
                                         pin_memory=False)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))
        print("train unlabel data size{}".format(config.batch_size * len(unlabel_data_loader)))
        return train_dataloader, unlabel_data_loader


def kon10k_8000(config):
    print("kon10k 8000 bullied successfully  " + str(config.split))
    if config.data_mode == 'train':
        train_data = dataset.train_dataset.train_data('./data/kon10k/train8000/train_' + str(config.split),
                                                      './data/kon10k/1024x768')
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))
        return train_dataloader

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('./data/kon10k/test8000/test_' + str(config.split),
                                                   './data/kon10k/1024x768')
        test_dataloader = DataLoader(test_data,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     drop_last=True,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(config.batch_size * len(test_dataloader)))
        return test_dataloader


def kon10k_2000(config):
    print("kon10k 2000 bullied successfully   " + str(config.split))
    if config.data_mode == 'train':
        train_data = dataset.train_dataset.train_data('./data/kon10k/train2000/labeled/labeled_' + str(config.split),
                                                      './data/kon10k/1024x768')
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=config.pin_memory)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))
        return train_dataloader

    if config.data_mode == 'test':
        test_data = dataset.test_dataset.test_data('./data/kon10k/test2000/test_' + str(config.split),
                                                   './data/kon10k/1024x768')
        test_dataloader = DataLoader(test_data,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     drop_last=True,
                                     num_workers=config.number_workers,
                                     pin_memory=False)
        print("test data size{}".format(config.batch_size * len(test_dataloader)))
        return test_dataloader

    if config.data_mode == 'semi':
        train_data = dataset.train_dataset.train_data(
            './data/kon10k/train2000/labeled/labeled_' + str(config.split),
            './data/kon10k/1024x768')
        unlabel_dataset = dataset.train_dataset.train_data(
            './data/kon10k/train2000/unlabeled/unlabeled_' + str(config.split),
            './data/kon10k/1024x768')
        train_dataloader = DataLoader(train_data,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=config.number_workers,
                                      pin_memory=False)
        unlabel_data_loader = DataLoader(unlabel_dataset,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         num_workers=config.number_workers,
                                         pin_memory=False)
        print("train data size{}".format(config.batch_size * len(train_dataloader)))
        print("train unlabel data size{}".format(config.batch_size * len(unlabel_data_loader)))
        return train_dataloader, unlabel_data_loader

#
# def kadid10k_1000(i, min_batch_size, train_mode):
#     print("kadid10k 1000 bullied successfully")
#     if train_mode == 'supervise':
#         train_data = dataset.train_dataset.train_data('./data/kadid10k/train1000/labeled/labeled_' + str(i) + '.txt',
#                                                       './data/kadid10k/images')
#         train_dataloader = DataLoader(train_data, batch_size=min_batch_size,
#                                       shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
#         print("train data size{}".format(min_batch_size * len(train_dataloader)))
#         return train_dataloader
#
#     if train_mode == 'test':
#         test_data = dataset.test_dataset.test_data('./data/kadid10k/test1000/test_' + str(i) + '.txt',
#                                                    './data/kadid10k/images')
#         test_dataloader = DataLoader(test_data, batch_size=min_batch_size,
#                                      shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
#         print("test data size{}".format(min_batch_size * len(test_dataloader)))
#         return test_dataloader
#
#     if train_mode == 'unsupervise':
#         train_data = dataset.train_dataset.train_data('./data/kadid10k/train1000/labeled/labeled_' + str(i) + '.txt',
#                                                       './data/kadid10k/images')
#         unlabel_dataset = dataset.train_dataset.train_data(
#             './data/kadid10k/train1000/unlabeled/unlabeled_' + str(i) + '.txt',
#             './data/kadid10k/images')
#         train_dataloader = DataLoader(train_data, batch_size=min_batch_size,
#                                       shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
#
#         unlabel_data_loader = DataLoader(unlabel_dataset, batch_size=min_batch_size * 3, shuffle=True,
#                                          drop_last=True, num_workers=4, pin_memory=True)
#         print("train data size{}".format(min_batch_size * len(train_dataloader)))
#         print("train unlabel data size{}".format(min_batch_size * len(unlabel_data_loader) * 3))
#         return train_dataloader, unlabel_data_loader
#
#
# def kadid10k_2000(i, min_batch_size, train_mode):
#     print("kadid10k 2000 bullied successfully")
#     if train_mode == 'supervise':
#         train_data = dataset.train_dataset.train_data('./data/kadid10k/train2000/labeled/labeled_' + str(i) + '.txt',
#                                                       './data/kadid10k/images')
#         train_dataloader = DataLoader(train_data, batch_size=min_batch_size,
#                                       shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
#         print("train data size{}".format(min_batch_size * len(train_dataloader)))
#         return train_dataloader
#
#     if train_mode == 'test':
#         test_data = dataset.test_dataset.test_data('./data/kadid10k/test2000/test_' + str(i) + '.txt',
#                                                    './data/kadid10k/images')
#         test_dataloader = DataLoader(test_data, batch_size=min_batch_size,
#                                      shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
#         print("test data size{}".format(min_batch_size * len(test_dataloader)))
#         return test_dataloader
#
#     if train_mode == 'unsupervise':
#         train_data = dataset.train_dataset.train_data('./data/kadid10k/train2000/labeled/labeled_' + str(i) + '.txt',
#                                                       './data/kadid10k/images')
#         unlabel_dataset = dataset.train_dataset.train_data(
#             './data/kadid10k/train1000/unlabeled/unlabeled_' + str(i) + '.txt',
#             './data/kadid10k/images')
#         train_dataloader = DataLoader(train_data, batch_size=min_batch_size,
#                                       shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
#
#         unlabel_data_loader = DataLoader(unlabel_dataset, batch_size=min_batch_size * 3, shuffle=True,
#                                          drop_last=True, num_workers=4, pin_memory=True)
#         print("train data size{}".format(min_batch_size * len(train_dataloader)))
#         print("train unlabel data size{}".format(min_batch_size * len(unlabel_data_loader) * 3))
#         return train_dataloader, unlabel_data_loader

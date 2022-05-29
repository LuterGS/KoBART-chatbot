import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from fastapi_app import MainFastAPI
from kobart_chit_chat import Base, ArgsBase, ChatDataModule, KoBARTConditionalGeneration

parser = argparse.ArgumentParser(description='KoBART chatbot-server')
parser.add_argument('--checkpoint_path', type=str, help='checkpoint path')
parser.add_argument('--chat', action='store_true', default=False, help='response generation on given user input')

if __name__ == "__main__":
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = ChatDataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = KoBARTConditionalGeneration(args)

    dm = ChatDataModule(args.train_file,
                        args.test_file,
                        os.path.join(args.tokenizer_path, 'model.json'),
                        max_seq_len=args.max_seq_len,
                        num_workers=args.num_workers)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=-1,
                                                       prefix='kobart_chitchat')
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])
    trainer.fit(model, dm)

    # launch fastAPI
    fastapi = MainFastAPI(model)
    fastapi.start_app()


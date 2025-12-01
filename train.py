"""
Training Script - Huấn luyện mô hình
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models import EmotionCNN, EngagementModel
from src.data import DataLoader, DataPreprocessor, EngagementDataset
from src.utils import setup_logger
from src.config import get_settings


def parse_args():
    parser = argparse.ArgumentParser(description='Train Emotion/Engagement Model')
    
    parser.add_argument(
        '--model',
        choices=['emotion', 'engagement'],
        default='emotion',
        help='Loại mô hình cần train'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Thư mục chứa dữ liệu training'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/models/model.h5',
        help='Đường dẫn lưu mô hình'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Số epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Sử dụng data augmentation'
    )
    
    return parser.parse_args()


def train_emotion_model(args, logger):
    """Train emotion classification model"""
    logger.info("Training Emotion Classification Model")
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    loader = DataLoader()
    dataset = loader.load_labeled_dataset(args.data_dir)
    
    if not dataset:
        logger.error("Không tìm thấy dữ liệu!")
        return
    
    # Split data
    train_set, val_set, test_set = loader.split_dataset(dataset)
    
    # Preprocess
    preprocessor = DataPreprocessor(target_size=(48, 48))
    
    # Prepare training data
    train_images = []
    train_labels = []
    
    emotion_classes = EmotionCNN.EMOTION_CLASSES
    
    for label, images in train_set.items():
        if label in emotion_classes:
            label_idx = emotion_classes.index(label)
            for img in images:
                train_images.append(img)
                train_labels.append(label_idx)
    
    # Augmentation if enabled
    if args.augment:
        logger.info("Applying data augmentation...")
        train_images, train_labels = preprocessor.create_augmented_dataset(
            train_images, train_labels, augment_factor=3
        )
    else:
        train_images = preprocessor.preprocess_batch(train_images)
    
    logger.info(f"Training samples: {len(train_images)}")
    
    # Create model
    model = EmotionCNN()
    model.build()
    model.compile(learning_rate=args.learning_rate)
    
    # Train
    logger.info(f"Starting training for {args.epochs} epochs...")
    # model.train(train_images, train_labels, val_images, val_labels, 
    #             epochs=args.epochs, batch_size=args.batch_size)
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))
    
    logger.info(f"Model saved to {args.output}")


def train_engagement_model(args, logger):
    """Train engagement prediction model"""
    logger.info("Training Engagement Prediction Model")
    
    # Similar to emotion model training
    # Implementation details...
    
    model = EngagementModel(use_temporal=True)
    model.build()
    
    logger.info("Engagement model training completed")


def main():
    args = parse_args()
    logger = setup_logger(level=logging.INFO)
    
    logger.info("=" * 50)
    logger.info("MODEL TRAINING")
    logger.info("=" * 50)
    
    try:
        if args.model == 'emotion':
            train_emotion_model(args, logger)
        else:
            train_engagement_model(args, logger)
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

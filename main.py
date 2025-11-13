from utils import get_data,train
from module import get_backbone,attch_projection_head
import argparse
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GroupShuffleSplit

parser = argparse.ArgumentParser(description='training setup')

parser.add_argument('--batch_size', type=int, default=1024, help='batch size of training')
parser.add_argument('--epoch', type=int, default=200, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--tem', type=float, default=1e-1, help='the hyperparameter temperature')
parser.add_argument('--dataset', type=str, default='ucihar', choices=['ucihar', 'motion', 'uschad'], help='dataset')
parser.add_argument('--backbone', type=str, default='tpn', help='backbone')
parser.add_argument('--p1', type=int, default=96, help='projection head dimension')
parser.add_argument('--p2', type=int, default=96, help='projection head dimension')
parser.add_argument('--p3', type=int, default=96, help='projection head dimension')
parser.add_argument('--cluster', type=str, default='birch', choices=['birch', 'kmeans'], help='cluster methods')
parser.add_argument('--cluster_num', type=int, default=8, help='cluster number')
def linear_evaluation(backbone, X_train, y_train, X_test, y_test, subjects_train, n_outputs, args):
    backbone.trainable = False
    eval_model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=backbone.input_shape[1:]),
        backbone,
        tf.keras.layers.Dense(n_outputs, activation='softmax')
    ])
    eval_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    y_train_1d_for_split = np.argmax(y_train, axis=1)
    train_sub_idx, val_idx = next(gss.split(X_train, y_train, groups=subjects_train))
    X_train_sub, y_train_sub = X_train[train_sub_idx], y_train[train_sub_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]
    
    val_subjects = np.unique(subjects_train[val_idx])
    checkpoint_filepath = '/kaggle/working/best_linear_model_fold.keras'
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0
    )
    es_callback = EarlyStopping(
        monitor='val_loss',
        patience=30,
        mode='min',
        restore_best_weights=True
    )
    eval_model.fit(
        X_train_sub, y_train_sub,
        batch_size=args.batch_size,
        epochs=300,
        validation_data=(X_val, y_val),
        callbacks=[cp_callback, es_callback],
        verbose=2
    )
    
    print("\n--- Test Set Evaluation Results ---")
    loss, accuracy = eval_model.evaluate(X_test, y_test, verbose=0)
    
    y_pred_probs = eval_model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Result:")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test F1-score (Macro): {f1 * 100:.2f}%")
    return f1
    
if __name__ == '__main__':
    args = parser.parse_args()
    all_results = []
    for group in range(1, 11):
        x_train, y_train, x_test, y_test, subjects_train  = get_data(args.dataset, group)
        print(f"DEBUG: x_train.shape = {x_train.shape}")
        print(f"DEBUG: y_train.shape = {y_train.shape}")
        n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    
        backbone = get_backbone(args.backbone,n_timesteps,n_features)
        model = attch_projection_head(backbone,args.p1,args.p2,args.p3)
        print("\n--- Starting Unsupervised Pre-training ---")
        train(model,x_train,args)
        print("\n--- Starting Linear Evaluation ---")
        _f1 = linear_evaluation(backbone, x_train, y_train, x_test, y_test, subjects_train, n_outputs, args)
        all_results.append({'group': group, 'f1_score': _f1})
    df_results = pd.DataFrame(all_results)
    avg_f1 = df_results['f1_score'].mean()
    print(f"\nAverage F1: {avg_f1 * 100:.2f}%")
    avg_row = pd.DataFrame([{'group': 'Average', 'f1_score': avg_f1}])
    final_df = pd.concat([df_results, avg_row], ignore_index=True)
    final_df.to_csv("results.csv", index=False)





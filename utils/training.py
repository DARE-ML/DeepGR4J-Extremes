import time
import numpy as np
import tensorflow as tf

# %%
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class Trainer:
    def __init__(self, model, optimizer, loss_fn=None, early_stopper=None):
        self.model = model
        self.optimizer = optimizer
        if isinstance(loss_fn, str):
            self.loss_fn = tf.keras.losses.get(loss_fn)
        elif isinstance(loss_fn, tf.keras.losses.Loss):
            self.loss_fn = loss_fn
        else:
            print("Invalid loss function. Using MeanSquaredError instead.")
            self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.early_stopper = early_stopper

        # Metrics
        self.train_loss_metric = tf.keras.metrics.MeanSquaredError()
        self.test_loss_metric = tf.keras.metrics.MeanSquaredError()
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            out = self.model(x, training=True)
            loss_value = self.loss_fn(y, out)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_loss_metric.update_state(y, out)
        return loss_value

    @tf.function
    def test_step(self, x, y):
        out = self.model(x, training=False)
        self.test_loss_metric.update_state(y, out)


    def train(self, train_ds, test_ds, args):
        
        # Construct dataloaders
        train_dl = train_ds.shuffle(140000).batch(args.batch_size)
        test_dl = test_ds.batch(args.batch_size)

        # Lists to store losses
        train_losses = []
        test_losses = []

        # Train the model
        for epoch in range(1, args.epochs+1):
            
            # Start time
            start_time = time.time()

            for step, batch_train in enumerate(train_dl):

                # Extract input and output batchs
                if args.ts_model == 'cnn':
                    batch_train['timeseries'] = tf.expand_dims(batch_train['timeseries'], axis=-1)
                x_batch_train = [batch_train['timeseries'] , batch_train['static']]
                y_batch_train = batch_train['target']

                # Train on batch
                loss_value = self.train_step(x_batch_train, y_batch_train)
                
                if args.verbose == 2 and step % 200 == 0:
                    print("Training loss at step %d: %.4f" % (step, loss_value.numpy()))
                    print("Seen so far: %s samples" % ((step + 1) * 128))
            
            # Update train metrics
            train_loss = self.train_loss_metric.result()
            self.train_loss_metric.reset_state()


            # Evaluate on validation set
            for batch_test in test_dl:
                if args.ts_model == 'cnn':
                    batch_test['timeseries'] = tf.expand_dims(batch_test['timeseries'], axis=-1)
                x_batch_val = [batch_test['timeseries'], batch_test['static']]
                y_batch_val = batch_test['target']
                self.test_step(x_batch_val, y_batch_val)
            
            # Update validation metrics
            test_loss = self.test_loss_metric.result()
            self.test_loss_metric.reset_state()
            
            if args.verbose == 1:
                print("\nEpoch %d" % (epoch,))
                print("Training loss over epoch: %.4f" % (train_loss.numpy()))
                print("Validation loss over epoch: %.4f" % (test_loss.numpy()))
                print("Time taken: %.2fs" % (time.time() - start_time))

            # Save losses
            train_losses.append(train_loss.numpy())
            test_losses.append(test_loss.numpy())
            
            if self.early_stopper is not None:
                if self.early_stopper.early_stop(test_loss.numpy()):
                    print(f"Training stopped early after {epoch} epochs!")
                    break

        return self.model, train_losses, test_losses

            
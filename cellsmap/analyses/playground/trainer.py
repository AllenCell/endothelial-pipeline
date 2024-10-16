import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from cellsmap.analyses.playground import const, viz
from torch.utils.data import DataLoader, random_split

class Trainer():

    def __init__(self, exp):
        self.exp = exp

    def set_dataset(self, dataset):
        train_dataset, test_dataset = Trainer.train_test_split(dataset)
        self._test_loader = DataLoader(test_dataset, batch_size=const.BATCH_SIZE, shuffle=False)
        self._train_loader = DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True)

    def train_test_split(dataset):
        test_len = int(len(dataset) * const.TEST_SIZE)
        train_len = len(dataset) - test_len
        return random_split(dataset, [train_len, test_len])
    
    def save_results(self, train_losses, test_losses, save_reconstructions=True):
        if save_reconstructions:
            viz.save_reconstruction_examples(self)
        viz.plot_loss_curves(self, train_losses, test_losses)

class TrainerAutoencoder(Trainer):

    def __init__(self, exp):
        super().__init__(exp)
        self._output_path = self.exp.get_output_dir()

    def train(self):

        criterion = nn.MSELoss()
        model = self.exp.get_architecture()
        optimizer = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
        
        train_losses, test_losses, best_test_loss = [], [], float('inf')
        
        for epoch in range(const.NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            for images in self._train_loader:
                images = images.to(self.exp.get_device())
                
                outputs = model(images)
                loss = criterion(outputs, images)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
            
            train_loss /= len(self._train_loader.dataset)
            train_losses.append(train_loss)
            
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for images in self._test_loader:
                    images = images.to(self.exp.get_device())
                    
                    outputs = model(images)
                    loss = criterion(outputs, images)
                    test_loss += loss.item() * images.size(0)
            
            test_loss /= len(self._test_loader.dataset)
            test_losses.append(test_loss)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), os.path.join(self._output_path, "weight.pth"))
                self.save_results(train_losses, test_losses)
                print("\tModel saved.")

            print(f"Epoch [{epoch+1}/{const.NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        self.save_results(train_losses, test_losses, save_reconstructions=False)

class TrainerBVAE(Trainer):

    def __init__(self, exp):
        super().__init__(exp)
        self._output_path = self.exp.get_output_dir()


    def train(self):

        model = self.exp.get_architecture()
        optimizer = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
        
        train_losses, test_losses, best_test_loss = [], [], float('inf')
        
        for epoch in range(const.NUM_EPOCHS):
            model.train()
            train_loss = 0
            for images in self._train_loader:
                images = images.to(self.exp.get_device())
                optimizer.zero_grad()
                recon_images, mu, logvar = model(images)
                
                recon_loss = F.mse_loss(recon_images, images, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + const.BETA * kld_loss
                
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            train_loss /= len(self._train_loader.dataset)
            train_losses.append(train_loss)
            
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for images in self._test_loader:
                    images = images.to(self.exp.get_device())
                    recon_images, mu, logvar = model(images)
                    recon_loss = F.mse_loss(recon_images, images, reduction='sum')
                    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + const.BETA * kld_loss
                    test_loss += loss.item()
            print(f"\tLast rec loss: {recon_loss:.2f}, KL loss: {kld_loss:.2f}.")

            test_loss /= len(self._test_loader.dataset)
            test_losses.append(test_loss)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), os.path.join(self._output_path, "weight.pth"))
                print("\tModel saved.")

            print(f"Epoch [{epoch+1}/{const.NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        self.save_results(train_losses, test_losses)

class TrainerPredictiveAutoencoder(Trainer):

    def __init__(self, exp):
        super().__init__(exp)
        self._output_path = self.exp.get_output_dir()

    def train(self):

        alpha = 20
        criterion = nn.MSELoss()
        class_criterion = nn.CrossEntropyLoss()
        model = self.exp.get_architecture()
        optimizer = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE)
        
        train_losses, test_losses, best_test_loss = [], [], float('inf')
        
        for epoch in range(const.NUM_EPOCHS):
            model.train()
            train_loss = 0.0
            for img_curr, img_next, label1, label2 in self._train_loader:

                label1 = label1.to(self.exp.get_device())
                label2 = label2.to(self.exp.get_device())
                img_curr = img_curr.to(self.exp.get_device())
                img_next = img_next.to(self.exp.get_device())

                optimizer.zero_grad()                
                output = model(img_curr)
                img_next_pred, logits1_pred, logits2_pred = output

                rec_loss = criterion(img_next, img_next_pred)# ~0.002
                label1_loss = class_criterion(logits1_pred, label1)# ~0.01
                label2_loss = class_criterion(logits2_pred, label2)# ~0.01
                loss = alpha * rec_loss + label1_loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(self._train_loader.dataset)
            train_losses.append(train_loss)
            
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for img_curr_test, img_next_test, label1_test, label2_test in self._test_loader:

                    label1_test = label1_test.to(self.exp.get_device())
                    label2_test = label2_test.to(self.exp.get_device())
                    img_curr_test = img_curr_test.to(self.exp.get_device())
                    img_next_test = img_next_test.to(self.exp.get_device())
                    
                    output_test = model(img_curr_test)
                    img_next_test_pred, logits1_test_pred, logits2_test_pred = output_test

                    rec_test_loss = criterion(img_next_test, img_next_test_pred)
                    label1_test_loss = class_criterion(logits1_test_pred, label1_test)# ~0.01
                    label2_test_loss = class_criterion(logits2_test_pred, label2_test)# ~0.01
                    loss = alpha*rec_test_loss + label1_test_loss
                    
                    test_loss += loss.item()
            
            test_loss /= len(self._test_loader.dataset)
            test_losses.append(test_loss)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), os.path.join(self._output_path, "weight.pth"))
                self.save_results(train_losses, test_losses)
                print("\tModel saved.")

            print(f"Epoch [{epoch+1}/{const.NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
            self.save_results(train_losses, test_losses, save_reconstructions=False)

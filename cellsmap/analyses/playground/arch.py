import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_size, nchannels=1):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Dynamically calculate how many downsampling layers are needed
        self.encoder = nn.Sequential(
            nn.Conv2d(nchannels, 16, kernel_size=3, stride=2, padding=1),  # Halve the size
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Halve again
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Halve again
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Halve again
            nn.ReLU()
        )

        # Calculate the size after the encoder's convolutional layers
        self.encoded_size = self._calculate_encoded_size(input_size)

        # Adjust the latent space based on the encoded size
        self.latent_space = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * self.encoded_size * self.encoded_size, latent_dim)  # Flatten into the latent space
        )

        # Decoder: reverse the encoding process
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.encoded_size * self.encoded_size),  # Map back to encoded space
            nn.ReLU(),
            nn.Unflatten(1, (128, self.encoded_size, self.encoded_size)),  # Unflatten into the spatial shape
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Double the size
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Double again
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Double again
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Double back to original size
            nn.Sigmoid()  # Sigmoid for pixel values in range [0, 1]
        )

        self.classifier_label1 = nn.Linear(latent_dim, 3)  # Classify in 3 classes
        self.classifier_label2 = nn.Linear(latent_dim, 3)  # Classify in 3 classes

    def _calculate_encoded_size(self, size):
        # Each Conv2d with stride=2 reduces size by half
        for _ in range(4):  # We have 4 Conv2d layers that halve the size
            size = (size - 1) // 2 + 1  # This handles odd sizes correctly
        return size

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.latent_space(encoded)
        decoded = self.decoder(latent)
        label1_out = torch.sigmoid(self.classifier_label1(latent))
        label2_out = torch.sigmoid(self.classifier_label2(latent))
        return decoded, label1_out, label2_out

class SmallConvAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_size, nchannels=1):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Smaller encoder with fewer filters
        self.encoder = nn.Sequential(
            nn.Conv2d(nchannels, 8, kernel_size=3, stride=2, padding=1),  # Halve the size
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Halve again
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Halve again
            nn.ReLU()
        )

        # Calculate the size after the encoder's convolutional layers
        self.encoded_size = self._calculate_encoded_size(input_size)

        # Latent space adjusted for smaller capacity
        self.latent_space = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * self.encoded_size * self.encoded_size, latent_dim)  # Reduced number of units
        )

        # Decoder with fewer filters to match encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * self.encoded_size * self.encoded_size),  # Map back to encoded space
            nn.ReLU(),
            nn.Unflatten(1, (32, self.encoded_size, self.encoded_size)),  # Unflatten into spatial shape
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Double the size
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # Double again
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Double back to original size
            nn.Sigmoid()  # Sigmoid for pixel values in range [0, 1]
        )

        self.classifier_label1 = nn.Linear(latent_dim, 3)  # Classify in 3 classes
        self.classifier_label2 = nn.Linear(latent_dim, 3)  # Classify in 3 classes

    def _calculate_encoded_size(self, size):
        # Each Conv2d with stride=2 reduces size by half
        for _ in range(3):  # We now have 3 Conv2d layers that halve the size
            size = (size - 1) // 2 + 1  # This handles odd sizes correctly
        return size

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.latent_space(encoded)
        decoded = self.decoder(latent)
        label1_out = torch.sigmoid(self.classifier_label1(latent))
        label2_out = torch.sigmoid(self.classifier_label2(latent))
        return decoded, label1_out, label2_out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BetaVAE(nn.Module):

    def __init__(self, latent_dim):
        super(BetaVAE, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc2 = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 256 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(h1.size(0), -1)
        return self.fc1(h1), self.fc2(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 256, 4, 4)
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class UNetAutoencoder(nn.Module):
    def __init__(self, latent_dim, input_size, nchannels=1):
        super().__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(nchannels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck (before latent space)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Calculate the size of the bottleneck feature map (H/16, W/16)
        self.encoded_size = self._calculate_encoded_size(input_size)

        # Fully connected layers for latent space
        self.fc1 = nn.Linear(256 * self.encoded_size * self.encoded_size, latent_dim)  # From bottleneck to latent dim
        self.fc2 = nn.Linear(latent_dim, 256 * self.encoded_size * self.encoded_size)  # From latent dim back to bottleneck

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Output layer
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def _calculate_encoded_size(self, size):
        # After 4 pooling layers, the size is divided by 16
        for _ in range(4):
            size = (size - 1) // 2 + 1
        return size

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc1_pooled = self.pool1(enc1)

        enc2 = self.enc2(enc1_pooled)
        enc2_pooled = self.pool2(enc2)

        enc3 = self.enc3(enc2_pooled)
        enc3_pooled = self.pool3(enc3)

        enc4 = self.enc4(enc3_pooled)
        enc4_pooled = self.pool4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc4_pooled)

        # Flatten bottleneck to pass through the fully connected layers
        bottleneck_flat = bottleneck.view(bottleneck.size(0), -1)  # Flatten to (batch_size, 256 * H/16 * W/16)
        
        # Latent space
        latent = self.fc1(bottleneck_flat)  # Map to latent dimension
        
        # Map back from latent dimension to bottleneck feature size
        bottleneck_flat = self.fc2(latent)
        
        # Reshape back to the bottleneck feature map shape (batch_size, 256, H/16, W/16)
        bottleneck = bottleneck_flat.view(bottleneck.size(0), 256, self.encoded_size, self.encoded_size)

        # Decoder
        up4 = self.upconv4(bottleneck)
        dec4 = torch.cat((up4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        up3 = self.upconv3(dec4)
        dec3 = torch.cat((up3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        up2 = self.upconv2(dec3)
        dec2 = torch.cat((up2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        up1 = self.upconv1(dec2)
        dec1 = torch.cat((up1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        out = self.out_conv(dec1)

        return out#, latent  # Return both output and latent variables

'''
1d645624: dts = [-5,-1,0,1], Noise:  No, Loss: classification
e89b993c: dts = [-5,-1,0,1], Noise:  No, Loss: reconstruction
8aea4053: dts = [-5,-1,0,1], Noise:  No, Loss: reconstruction + classification
acedab28: dts = [ 0, 0,0,1], Noise:  No, Loss: reconstruction + classification
1b7035ac: dts = [ 0, 0,0,1], Noise: 0.5, Loss: reconstruction + classification
f117299c: dts = [ 0, 0,0,0], Noise: 0.5, Loss: reconstruction + classification
'''

from cellsmap.util import io
from cellsmap.analyses.playground import const, trans, dataset, pipeline, arch, trainer

if __name__ == '__main__':

    reader = io.load_dataset("20240305_T01_001", channels=["CDH5_Tubulin", "Nuc_Seg"], time_start=0, level=2)
    print(reader.shape)

    with pipeline.run_experiment(debug=False) as exp:

        ptrans = trans.PairedTransform(add_noise=True)

        dataset = dataset.ZARRTemporalSignalCenteredDataset(reader=reader, dts=[0,0,0,0], transform=ptrans, debug=exp.get_debug())
        exp.set_architecture(arch.SmallConvAutoencoder(latent_dim=const.N, input_size=const.M, nchannels=3))
        trainer = trainer.TrainerPredictiveAutoencoder(exp)
        trainer.set_dataset(dataset)
        trainer.train()

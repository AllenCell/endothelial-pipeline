from cellsmap.util import io
from cellsmap.analyses.playground import const, trans, dataset, pipeline, arch, trainer

if __name__ == '__main__':

    reader = io.load_dataset("20240305_T01_001", channels=["CDH5_Tubulin", "Nuc_Seg"], time_start=0, level=2)
    print(reader.shape)

    with pipeline.cellsmap_dl_experiment(debug=False) as exp:

        dataset = dataset.ZARRDataset(reader=reader, transform=trans.transform, debug=exp.get_debug())
        exp.set_architecture(arch.BetaVAE(latent_dim=const.N))
        trainer = trainer.TrainerBVAE(exp)
        trainer.set_dataset(dataset)
        trainer.train()

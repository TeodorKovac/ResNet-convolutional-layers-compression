# ResNet-convolutional-layers-compression
ResNet convolutional layers compression with the use of CP decomposition.


Skripty určené ke kompresy konvolučních vrstev residuální neuronové sítě ResNet18.

Konvoluční vrstvy obsahují velké množství dat, bývají nejobjemnější součástí dané sítě a proto jsme se zaměřovali na kompresy právě tohoto prvku.

Síť ResNet18 je volně dostupná ke stažení na internetu. Síť je často využívána pro úlohy rozpoznávání obrazu. Aby bylo vůbec možné vyhodnocovat úspěšnost komprese, je třeba provést tzv. Transfer Learning. Nebo-li před předtrénovanou síť upravit pro vyhodnocování příkladů z nějaké dostupné databáze. Tento úkon je prováděn ve scriptu 'TransferLearning_resnet.m'.

Jednotlivé konvoluční vrstvy neuronové sítě bývají reprezentovány tenzorem čtvrtého řádu a proto je možné na ně aplikovat metody tenzorového rozkladu. Funkce 'CPDlayerReplace()' původní konvoluční vrstvu sítě čtyřmi podstatně menšími konvolučními vrstvami, kde váhy jsou faktorové matice původního tenzoru nalezené rozkladovou metodou KLM (Krylov-Levenberg-Marquardt).

Script 'ResNet18_script.m' postupně prochází jednotlivé konvoluční vrstvy sítě ResNet18 a zaměňuje je na paměť méně náročnými vrstvami z faktorových matic tenzorového rozkladu.

Poznámka: Tento depositář má sloužit jen jako rámcová ukázka implementace řešení daného problému.

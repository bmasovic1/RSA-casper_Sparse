# README - Pokretanje programa

## Naziv:
casper_Sparse.exe

## Sintaksa:
```bash
casper_Sparse.exe M K N sparsity(%)
```

## Parametri:
*   **M** - broj redova sparse matrice A
*   **K** - broj kolona sparse matrice A (u isto vrijeme broj redova dense matrice B)
*   **N** - broj kolona dense matrice B
*   **sparsity(%)** - procenat popunjenosti koji program koristi za generisanje sparse matrice (npr. 1 = ~1%)

## Dimenzije matrica:
A je sparse matrica dimenzija M x K  
B je dense matrica dimenzija K x N  
(rezultat mnozenja je C dimenzija M x N)

## Primjer:
```bash
casper_Sparse.exe 1000 1000 1000 1
```

## Napomena:
Pokrenuti iz Command Prompt/PowerShell-a u folderu gdje se nalazi casper_Sparse.exe.

# rei201verkefni1
verkefni1

Hlekkur að námskeiðinu:

* https://notendur.hi.is/~pmelsted/kennsla/rei/

## Uppsetning

Þarft að sækja 

* https://notendur.hi.is/~pmelsted/kennsla/rei/albums.zip
* https://notendur.hi.is/~pmelsted/kennsla/rei/mnist.zip

og sprengja skrárnar.  Einnig þarf að búa til möppuna `_ignore`.

Eða keyrðu `init.sh` ef þú ert með Bash og Curl á tölvunni þinni.

## Búa til gögn

### OS X

Keyrðu annaðhvort `generate_data.ipynb` Jupyter eða `generate_data_standalone.py` með Python 3, eins og

```bash
$ python3 generate_data_standalone.py
```

Niðurstöðunar vistast í möppuna `_ignore`.  Reglulega eru skrárnar í `_ignore` síaðar, þ.e.a.s. skrár sem gefa bestu niðurstöðu halda áfram að vera til staðar en hinar eru eyddar.

### Windows

Ábyggilega best að nota Jupyter hér, annars veit ég ekki.

import pr_utility as pru
import pr_augment as pra

path = 'sourceimages/test_binary.png'

pra.interactive_threshold(path)
pru.emb_val(path, pru.find_emb(path))

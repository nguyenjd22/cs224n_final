import os; import psutil; import timeit
from nlp import load_dataset
from nlp import DownloadConfig
mem_before = psutil.Process(os.getpid()).memory_info().rss >> 20
conf = DownloadConfig(cache_dir='/data5/nguyenjd/wikipedia')
wiki = load_dataset("wikipedia", "20200501.en", split='train', cache_dir='/data5/nguyenjd/wikipedia', download_config=conf)
#wiki = load_dataset("wikipedia", "20200501.en", split='train')
mem_after = psutil.Process(os.getpid()).memory_info().rss >> 20
mem_after = psutil.Process(os.getpid()).memory_info().rss >> 20
print(f"RAM memory used: {(mem_after - mem_before)} MB")

s = """batch_size = 1000
for i in range(0, len(wiki), batch_size):
    batch = wiki[i:i + batch_size]
"""
#print(wiki[0])
#print(wiki[1])
time = timeit.timeit(stmt=s, number=1, globals=globals())
size = wiki.dataset_size / 2**30
print(f"Iterated over the {size:.1f} GB dataset in {time:.1f} s, i.e. {size * 8/time:.1f} Gbit/s")

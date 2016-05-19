## modified from source https://github.com/praveen97uma/benchmark

import os
import random
import time
import platform
import psutil

from os.path import getsize, join
from matplotlib import pyplot
from memory_profiler import memory_usage

UNIX_WORDS_DICT = '/usr/share/dict/words'
MAX_WORD_LENGTH = 10
MAX_WORDS_IN_TEXT = 20
MAX_INDEX_ENTRIES = 1000


def generate_word(length, words_list):
    random_word = random.choice(words_list)
    random_word = random_word[:-1]
    if len(random_word) > length:
        return generate_word(length, words_list)
    return random_word


def get_words_list(words_file_path=UNIX_WORDS_DICT):
    import codecs
    f = codecs.open(words_file_path, encoding='utf-8').readlines()
    return f


def generate_text(words_list, max_words=MAX_WORDS_IN_TEXT,
                  max_word_length=MAX_WORD_LENGTH,
                  delim=" "):
    text = [generate_word(max_word_length, words_list) for _ in xrange(max_words)]
    return delim.join(text)


def generate_keywords(max_word_length=MAX_WORD_LENGTH, no_of_keywords=100):
    words_list = get_words_list()
    keywords = [generate_word(max_word_length, words_list) for _ in xrange(no_of_keywords)]
    return keywords


def generate_data():
    words_list = get_words_list()
    documents = []

    for i in xrange(MAX_INDEX_ENTRIES):
        title = generate_text(words_list)
        slug = generate_text(words_list, max_words=5)
        description = generate_text(words_list, max_words=15)
        id = i
        documents.append({
            'id': i,
            'title': title,
            'slug': slug,
            'description': description,
        })
        # print "%d \n Title: %s \n Slug: %s\n Description: %s"%(i, title, slug, description)
    return documents


def timer(func, *args, **kwargs):
    timer_start = time.time()
    func(*args, **kwargs)
    timer_stop = time.time()
    return timer_stop - timer_start


def memory_consumption(func, args=(), kwargs={}):
    memory_used = memory_usage((func, args, kwargs))
    average_memory_used = sum(memory_used) / len(memory_used)
    return average_memory_used


def get_dir_size(top):
    total_size = 0
    for root, dirs, files in os.walk(top):
        total_size += sum([getsize(join(root, name)) for name in files])
    return float(total_size) / (1024 * 1024)


def get_file_size(path):
    try:
        file_size = getsize(path)
    except os.error:
        file_size = 0
    return float(file_size) / (1024 * 1024)


def plot(X, Y, x_label=None, y_label=None, *args, **kwargs):
    pyplot.plot(X, Y, *args, **kwargs)
    if x_label:
        pyplot.xlabel(x_label)
    if y_label:
        pyplot.ylabel(y_label)

    pyplot.show()


def printPlatformInfo():
    print "--------------------------------------------------------------------"
    print "Platform: %s " % (platform.platform())
    print "Python Version: %s, Compiler: %s" % (platform.python_version(),
                                                platform.python_compiler())
    print "Number of CPUs: %d, Total Physical Memory: %f MB" % (psutil.NUM_CPUS,
                                                                psutil.TOTAL_PHYMEM / (1024.0 * 1024.0))
    print "--------------------------------------------------------------------"


def printSoftwareInfo():
    import whoosh, psutil, matplotlib
    whoosh_version = whoosh.versionstring()
    psutil_version = psutil.__version__
    matplotlib_version = matplotlib.__version__
    python_version = platform.python_version()
    print "Software Tools used:\n ",
    print "Whoosh %s, psutil %s, matplotlib %s, python %s" % (whoosh_version,
                                                              psutil_version,
                                                              matplotlib_version,
                                                              python_version)


if __name__ == '__main__':
    from whoosh_bench import WHOOSH_INDEX_DIR

    time_taken = timer(generate_data)
    memory_used = memory_consumption(generate_data)
    print printPlatformInfo()

    print "Time : ", time_taken
    print "Memory: ", memory_used
    print "Size: ", get_dir_size(WHOOSH_INDEX_DIR)
    print "size of file", get_file_size('whoosh_bench.py')


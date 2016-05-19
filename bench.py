## modified from source https://github.com/praveen97uma/benchmark
## I will use this as example

import os
import sys
import cPickle as pickle
import shutil
import time

import whoosh
from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.index import open_dir, create_in
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.filedb.multiproc import MultiSegmentWriter

from matplotlib import pyplot

import bench_utils

OUTPUT_TO_FILE = True
CURRENT_DIR = os.path.dirname(__file__)
WHOOSH_INDEX_DIR = os.path.join(CURRENT_DIR, 'whoosh_index')
DOCUMENTS_DB_PATH = os.path.join(CURRENT_DIR, 'documents.db')
MAX_NO_OF_KEYWORDS = 10

if OUTPUT_TO_FILE:
    local_time = time.localtime()
    time_format = "%(tm_year)s-%(tm_mon)s-%(tm_mday)s-%(tm_hour)s-%(tm_min)s-%(tm_sec)s"
    time_dict = {
        'tm_year': local_time.tm_year,
        'tm_mon': local_time.tm_mon,
        'tm_mday': local_time.tm_mday,
        'tm_hour': local_time.tm_hour,
        'tm_min': local_time.tm_min,
        'tm_sec': local_time.tm_sec,
    }
    current_time = time_format % (time_dict)
    f = open("Benchmark_whoosh_%s.txt" % (current_time), "wb")
    sys.stdout = f


def get_documents():
    documents = bench_utils.generate_data()
    pickle.dump(documents, open(DOCUMENTS_DB_PATH, "wb"))

    return documents


def create_index(use_multiprocessing=False):
    schema_fields = {
        'id': NUMERIC(stored=True),
        'slug': TEXT,
        'title': TEXT,
        'description': TEXT,
    }

    schema = Schema(**schema_fields)

    if os.path.exists(WHOOSH_INDEX_DIR):
        shutil.rmtree(WHOOSH_INDEX_DIR)
    os.mkdir(WHOOSH_INDEX_DIR)

    ix = create_in(WHOOSH_INDEX_DIR, schema)
    if use_multiprocessing:
        writer = MultiSegmentWriter(ix, limitmb=128)
    else:
        writer = ix.writer(limitmb=256)

    documents = get_documents()

    for doc in documents:
        writer.add_document(**doc)
    writer.commit()
    ix.close()


def simple_search(query):
    ix = open_dir(WHOOSH_INDEX_DIR)
    with ix.searcher() as searcher:
        query_string = QueryParser('title', ix.schema).parse(query)
        results = searcher.search(query_string)
        for result in results:
            obj = repr(result.fields())


def complex_search(query):
    ix = open_dir(WHOOSH_INDEX_DIR)
    with ix.searcher() as searcher:
        query_string = MultifieldParser(['title', 'slug', 'description'], ix.schema).parse(query)
        results = searcher.search(query_string)
        for result in results:
            obj = repr(result.fields())


def create_index_benchmark(verbose=True, use_multiprocessing=False):
    time_taken = bench_utils.timer(create_index, use_multiprocessing)
    memory_used = bench_utils.memory_consumption(create_index, (), {'use_multiprocessing': use_multiprocessing})
    index_size = bench_utils.get_dir_size(WHOOSH_INDEX_DIR)

    if verbose:
        print "\n===== Performance for index creation ====="
        print "No. of indexed documents: %d" % (bench_utils.MAX_INDEX_ENTRIES)
        print "No. of words in each document: %d" % (bench_utils.MAX_WORDS_IN_TEXT)
        print "Length of each word: %d chars" % (bench_utils.MAX_WORD_LENGTH)
        print "Average time taken: %f secs" % (time_taken)
        print "Average memory used: %f MB" % (memory_used)

    return (time_taken, memory_used, index_size)


def multiple_create_index_benchmarks(rang=1001, use_multiprocessing=False):
    print "\n===== Performace of index creation ====="
    print "Multiprocessing : %s" % (["No", "Yes"][int(use_multiprocessing)])
    print "No. of words in each document: %d" % (bench_utils.MAX_WORDS_IN_TEXT)
    print "Length of each word: %d chars\n" % (bench_utils.MAX_WORD_LENGTH)
    print "No. of Docs  Time(secs)    Memory(MB)      Index Size(MB)"
    print "----------------------------------------------------------"

    start = 100
    steps = 100
    X = range(start, rang, steps)
    Y = []
    for no_of_docs in X:
        bench_utils.MAX_INDEX_ENTRIES = no_of_docs
        (time_taken, memory_used, index_size) = create_index_benchmark(False, use_multiprocessing)
        Y.append(time_taken)
        print "%-10d %10f     %10f     %10f" % (no_of_docs, time_taken, memory_used, index_size)

    return (X, Y)
    bench_utils.plot(X, Y, x_label="No. of Docs", y_label="Time taken to index")


def simple_search_benchmarks(gen_index=False, no_of_docs=1000):
    if gen_index:
        bench_utils.MAX_INDEX_ENTRIES = no_of_docs
        create_index()

    keywords = bench_utils.generate_keywords(no_of_keywords=MAX_NO_OF_KEYWORDS)
    index_size = bench_utils.get_dir_size(WHOOSH_INDEX_DIR)

    print "\n===== Performance of searching of simple queries ====="
    print "Size of the index: %f MB" % (index_size)
    print "No. of indexed documents: %d " % (bench_utils.MAX_INDEX_ENTRIES)
    print "No of search queries: %d\n" % (len(keywords))
    print "-------------------------------------------------------------------"
    print "Search Word       Time(sec)     Memory(MB)"
    print "----------------------------------------------"

    time_taken = 0
    memory_used = 0

    for word in keywords:
        tt = bench_utils.timer(simple_search, word)
        mu = bench_utils.memory_consumption(simple_search, (word,))
        time_taken += tt
        memory_used += mu
        print "%-10s      %10f     %10f" % (word, tt, mu)

    avg_time = time_taken / len(keywords)
    avg_memory = memory_used / len(keywords)

    print "\nAverage time taken: %f secs" % (avg_time)
    print "Average memory used: %f MB" % (avg_memory)


def complex_search_benchmarks(gen_index=False, no_of_docs=1000):
    if gen_index:
        bench_utils.MAX_INDEX_ENTRIES = no_of_docs
        create_index()

    keywords = bench_utils.generate_keywords(no_of_keywords=MAX_NO_OF_KEYWORDS)
    index_size = bench_utils.get_dir_size(WHOOSH_INDEX_DIR)

    print "\n===== Performance of searching of complex queries in Whoosh ====="
    print "Size of the index: %f MB" % (index_size)
    print "No. of indexed documents: %d " % (bench_utils.MAX_INDEX_ENTRIES)
    print "No of search queries: %d" % (len(keywords))
    print "------------------------------------------------------------------"
    print "Search Word       Time(sec)     Memory(MB)"
    print "----------------------------------------------"

    time_taken = 0
    memory_used = 0

    for word in keywords:
        tt = bench_utils.timer(complex_search, word)
        mu = bench_utils.memory_consumption(complex_search, (word,))
        time_taken += tt
        memory_used += mu
        print "%-10s      %10f     %10f" % (word, tt, mu)

    avg_time = time_taken / len(keywords)
    avg_memory = memory_used / len(keywords)

    print "\nAverage time taken: %f secs" % (avg_time)
    print "Average memory used: %f MB" % (avg_memory)


def run_index_benchmarks():
    no_of_documents = 500
    (x, y1) = multiple_create_index_benchmarks(no_of_documents, use_multiprocessing=True)
    (x, y2) = multiple_create_index_benchmarks(no_of_documents, use_multiprocessing=False)
    """
    fig = pyplot.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(x, y1)
    ax1.set_ylabel("Index creation time using multiprocessing")
    ax2=ax1.twinx()
    ax2.plot(x, y2, 'r')
    ax2.set_ylabel("Index creating time without using multiprocessing")
    ax2.set_xlabel("No. of documents indexed")
    pyplot.show()
    """
    pyplot.xlabel('No. of Docs')
    pyplot.ylabel('Indexing time')
    pyplot.title(r'Indexing time with multiprocessing and without multiprocessing')
    pyplot.plot(x, y1, label=r'with multiprocessing', color='red')
    pyplot.plot(x, y2, label=r'without multiprocessing')
    pyplot.legend(loc='upper right')

    pyplot.savefig("Index_creation.jpg")


def run_search_benchmarks():
    simple_search_benchmarks(gen_index=True, no_of_docs=500)
    complex_search_benchmarks(gen_index=True, no_of_docs=500)


if __name__ == '__main__':
    bench_utils.printPlatformInfo()
    bench_utils.printSoftwareInfo()
    run_search_benchmarks()
    # run_index_benchmarks()
    # run_complex_search_benchmarks(gen_index=False, no_of_docs=100)

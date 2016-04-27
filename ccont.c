/*
 * ccont - cache line contention measurement tool
 *
 *   Copyright (C) 2016 Roman Pen <r.peniaev@gmail.com>
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Description:
 *   The goal is to measure cache contention on different NUMA nodes,
 *   burn different CPUs, execute different instructions with different
 *   load patterns, e.g. the following is the list of three load patterns
 *   which were executed on machine with 2 NUMA nodes and 8 CPUs:
 *
 *   o cpu-increase - on each iteration number of CPU is increased:
 *
 *     Nodes  N0   N1  CPUs    operation       min       max       avg     stdev
 *      CPUs *--- ----    1      cmpxchg     8.938     8.938     8.938     0.000
 *      CPUs **-- ----    2      cmpxchg    36.114    36.119    36.117     0.004
 *      CPUs ***- ----    3      cmpxchg    54.270    54.272    54.271     0.001
 *      CPUs **** ----    4      cmpxchg    72.292    72.321    72.313     0.013
 *      CPUs **** *---    5      cmpxchg    61.691   108.060    98.782    20.735
 *      CPUs **** **--    6      cmpxchg   101.316   136.923   125.059    18.369
 *      CPUs **** ***-    7      cmpxchg   151.639   169.218   161.702     9.358
 *      CPUs **** ****    8      cmpxchg   192.281   196.250   194.281     2.098
 *
 *   o node-cascade - on each iteration CPUs from each node are burned:
 *
 *     Nodes  N0   N1  CPUs    operation       min       max       avg     stdev
 *      CPUs **** ----    4      cmpxchg    72.287    72.322    72.310     0.016
 *      CPUs ---- ****    4      cmpxchg    72.327    72.333    72.330     0.003
 *
 *   o cpu-rollover - on each iteration executor thread rolls to another CPU on
 *   the next node, keeping the same amount of CPUs burning:
 *
 *     Nodes  N0   N1  CPUs    operation       min       max       avg     stdev
 *      CPUs **** ----    4      cmpxchg    48.769    48.774    48.772     0.002
 *      CPUs ***- *---    4      cmpxchg    85.506    97.754    94.683     6.118
 *      CPUs **-- **--    4      cmpxchg   116.803   121.450   119.108     2.658
 *      CPUs *--- ***-    4      cmpxchg    91.312   103.877   100.721     6.273
 *      CPUs ---- ****    4      cmpxchg    48.288    48.368    48.323     0.038
 *
 *   Memory chunk for each load is always allocated on the node#0.
 *
 * How to build:
 *     gcc -o ccont ccont.c -O2 -lpthread -lnuma -lm -lrt -Wall
 */

#define _GNU_SOURCE
#include <getopt.h>
#include <stdio.h>
#include <unistd.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <time.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <numa.h>
#include <numaif.h>

#define  _10_MLN 10000000
#define   _1_MLN 1000000
#define _100_THD 100000
#define  _10_THD 10000

#define UNROLL_LOOP
#define SPINS  _10_MLN

#ifdef UNROLL_LOOP
#define MULT       _10_THD
#define UNROLL(fn) for_1000(fn)
#else
#define MULT       SPINS
#define UNROLL(fn) fn
#endif

#define for_10(fn)     fn; fn; fn; fn; fn; fn; fn; fn; fn; fn;
#define for_100(fn)    for_10(fn); for_10(fn); for_10(fn); for_10(fn);		\
		       for_10(fn); for_10(fn); for_10(fn); for_10(fn);		\
		       for_10(fn); for_10(fn);
#define for_1000(fn)   for_100(fn); for_100(fn); for_100(fn); for_100(fn);	\
		       for_100(fn); for_100(fn); for_100(fn); for_100(fn);	\
		       for_100(fn); for_100(fn);

#define BURN_CPU(func) ({			\
	int i;					\
						\
	for (i = 0; i < MULT; i++) {		\
		UNROLL(func);			\
	}					\
})

#define __cache_line_aligned __attribute__((__aligned__(64)))
#define ARRAY_LENGTH(a) (sizeof(a)/sizeof((a)[0]))
#define __compiletime_error(message) __attribute__((error(message)))

#define barrier() asm volatile ("" ::: "memory")

typedef uint64_t u64;
typedef int64_t s64;

typedef uint32_t u32;
typedef int32_t s32;

typedef uint16_t u16;
typedef int16_t s16;

typedef uint8_t u8;
typedef int8_t s8;

struct thr_param;
typedef void (*burn_fn_t)(struct thr_param *arg);

#define CPU_MAX  512
#define DIST_MAX 8

typedef unsigned long data_t;
typedef data_t data_arr_t[CPU_MAX * DIST_MAX];

struct param {
	int                lock;
	int                ready;
	int                dist;
	struct bitmask     *cpum;
	burn_fn_t          burn_fn;
	data_t             *data;
	unsigned long long ns[CPU_MAX];
};

struct thr_param {
	struct param *p;
	unsigned cpu;
};

extern void __cmpxchg_wrong_size(void)
	__compiletime_error("Bad argument size for cmpxchg");

static inline int atomic_inc(int *v)
{
	return __sync_fetch_and_add(v, 1);
}

static inline void lin_atomic_inc(int *v)
{
	asm volatile("\n\tlock; incl %0"
		     : "+m" (v));
}

static inline int atomic_read(int *v)
{
	return *(volatile int *)v;
}

static inline int test_bit(int nr, const void *addr)
{
	unsigned char v;
	const unsigned int *p = (const unsigned int *)addr;

	asm("btl %2,%1; setc %0" : "=qm" (v) : "m" (*p), "Ir" (nr));
	return v;
}

static inline void set_bit(int nr, void *addr)
{
	asm("btsl %1,%0" : "+m" (*(unsigned int *)addr) : "Ir" (nr));
}

/*
 * Atomic compare and exchange.  Compare OLD with MEM, if identical,
 * store NEW in MEM.  Return the initial value in MEM.  Success is
 * indicated by comparing RETURN with OLD.
 */
#define __raw_cmpxchg(ptr, old, new, size, lock)			\
({									\
	__typeof__(*(ptr)) __ret;					\
	__typeof__(*(ptr)) __old = (old);				\
	__typeof__(*(ptr)) __new = (new);				\
	switch (size) {							\
	case 1:						\
	{								\
		volatile u8 *__ptr = (volatile u8 *)(ptr);		\
		asm volatile(lock "cmpxchgb %2,%1"			\
			     : "=a" (__ret), "+m" (*__ptr)		\
			     : "q" (__new), "0" (__old)			\
			     : "memory");				\
		break;							\
	}								\
	case 2:						\
	{								\
		volatile u16 *__ptr = (volatile u16 *)(ptr);		\
		asm volatile(lock "cmpxchgw %2,%1"			\
			     : "=a" (__ret), "+m" (*__ptr)		\
			     : "r" (__new), "0" (__old)			\
			     : "memory");				\
		break;							\
	}								\
	case 4:						\
	{								\
		volatile u32 *__ptr = (volatile u32 *)(ptr);		\
		asm volatile(lock "cmpxchgl %2,%1"			\
			     : "=a" (__ret), "+m" (*__ptr)		\
			     : "r" (__new), "0" (__old)			\
			     : "memory");				\
		break;							\
	}								\
	case 8:						\
	{								\
		volatile u64 *__ptr = (volatile u64 *)(ptr);		\
		asm volatile(lock "cmpxchgq %2,%1"			\
			     : "=a" (__ret), "+m" (*__ptr)		\
			     : "r" (__new), "0" (__old)			\
			     : "memory");				\
		break;							\
	}								\
	default:							\
		__cmpxchg_wrong_size();					\
	}								\
	__ret;								\
})

#define LOCK_PREFIX "\n\tlock; "

#define __cmpxchg(ptr, old, new, size)					\
	__raw_cmpxchg((ptr), (old), (new), (size), LOCK_PREFIX)

#define cmpxchg(ptr, old, new)					\
	__cmpxchg(ptr, old, new, sizeof(*(ptr)))

static inline unsigned long xchg(unsigned long *p, unsigned long val)
{
	return __atomic_exchange_n(p, val, __ATOMIC_SEQ_CST);
}

static inline unsigned long long rdtsc(void)
{
	unsigned int low, high;

	asm volatile ("mfence" ::: "memory");
	asm volatile("rdtsc" : "=a" (low), "=d" (high));

	return low | ((unsigned long long)high) << 32;
}

static inline unsigned long long usecs(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return ((unsigned long long)tv.tv_sec * 1000000ull) + tv.tv_usec;
}

static inline unsigned long long nsecs(void)
{
    struct timespec ts = {0, 0};

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((unsigned long long)ts.tv_sec * 1000000000ull) + ts.tv_nsec;
}

static void burn_cpu__idle_loop(struct thr_param *arg)
{
	BURN_CPU(barrier());
}

static void burn_cpu__memset64(struct thr_param *arg)
{
	data_t *d;

	d = arg->p->data;
	BURN_CPU(memset(d, 0, 64));
}

static void burn_cpu__memset128(struct thr_param *arg)
{
	data_t *d;

	d = arg->p->data;
	BURN_CPU(memset(d, 0, 128));
}

static void burn_cpu__memset256(struct thr_param *arg)
{
	data_t *d;

	d = arg->p->data;
	BURN_CPU(memset(d, 0, 256));
}

static void burn_cpu__test_bit(struct thr_param *arg)
{
	int bit;
	data_t *d;

	bit = arg->cpu << arg->p->dist;
	d = arg->p->data;
	BURN_CPU(test_bit(bit, d));
}

static void burn_cpu__set_bit(struct thr_param *arg)
{
	int bit;
	data_t *d;

	bit = arg->cpu << arg->p->dist;
	d = arg->p->data;
	BURN_CPU(set_bit(bit, d));
}

static void burn_cpu__xadd(struct thr_param *arg)
{
	int ind;
	data_t *d;

	ind = arg->cpu * arg->p->dist;
	d = &arg->p->data[ind];
	BURN_CPU(atomic_inc((int *)d));
}

static void burn_cpu__inc(struct thr_param *arg)
{
	int ind;
	data_t *d;

	ind = arg->cpu * arg->p->dist;
	d = &arg->p->data[ind];
	BURN_CPU(lin_atomic_inc((int *)d));
}

static void burn_cpu__cmpxchg(struct thr_param *arg)
{
	int ind;
	data_t *d;

	ind = arg->cpu * arg->p->dist;
	d = &arg->p->data[ind];
	BURN_CPU(cmpxchg(d, 0, 1));
}

static void burn_cpu__xchg(struct thr_param *arg)
{
	int ind;
	data_t *d;

	ind = arg->cpu * arg->p->dist;
	d = &arg->p->data[ind];
	BURN_CPU(xchg(d, 0));
}

static void burn_cpu__mfence(struct thr_param *arg)
{
	BURN_CPU(asm volatile ("mfence" ::: "memory"));
}

static void burn_cpu__mov_mfence(struct thr_param *arg)
{
	int ind;
	data_t *d;

	ind = arg->cpu * arg->p->dist;
	d = &arg->p->data[ind];
	BURN_CPU(*d = 0;				\
		 asm volatile ("mfence" ::: "memory"));
}

static void burn_cpu__lfence(struct thr_param *arg)
{
	BURN_CPU(asm volatile ("lfence" ::: "memory"));
}

static void burn_cpu__sfence(struct thr_param *arg)
{
	BURN_CPU(asm volatile ("sfence" ::: "memory"));
}

static void burn_cpu__syscall(struct thr_param *arg)
{
	/* Invalid syscall to be able return from the kernel immediately */
	BURN_CPU(syscall(99999));
}

struct burn_ent {
	const char    *name;
	const char    *desc;
	burn_fn_t     fn;
} const burn_table[] = {
	{"idle",      "Idle loop, for calibration.",
	 burn_cpu__idle_loop},
	{"memset64",  "Memset of 64 bytes.",
	 burn_cpu__memset64},
	{"memset128", "Memset of 128 bytes.",
	 burn_cpu__memset128},
	{"memset256", "Memset of 256 bytes.",
	 burn_cpu__memset256},
	{"test_bit",  "Atomically tests bit.",
	 burn_cpu__test_bit},
	{"set_bit",   "Atomically sets bit.",
	 burn_cpu__set_bit},
	{"xadd",      "Atomically increments value, __sync_fetch_and_add().",
	 burn_cpu__xadd},
	{"inc",       "Atomically increments value, 'inc' instruction.",
	 burn_cpu__inc},
	{"cmpxchg",   "Atomic compare and xchange.",
	 burn_cpu__cmpxchg},
	{"xchg",      "Atomic xchange.",
	 burn_cpu__xchg},
	{"mfence",    "'mfence' instruction.",
	 burn_cpu__mfence},
	{"mov+mfence", "'mov+mfence' instructions.",
	 burn_cpu__mov_mfence},
	{"lfence",    "'lfence' instruction.",
	 burn_cpu__lfence},
	{"sfence",    "'sfence' instruction.",
	 burn_cpu__sfence},
	{"syscall",    "To kernel and back.",
	 burn_cpu__syscall},
};

static void self_setaffinity(unsigned cpu)
{
	cpu_set_t cpuset;
	int err;

	CPU_ZERO(&cpuset);
	CPU_SET(cpu, &cpuset);
	err = pthread_setaffinity_np(pthread_self(),
				     sizeof(cpu_set_t),
				     &cpuset);
	assert(err == 0);
	assert(cpu == sched_getcpu());
}

static void check_memory_node_for_addr(void* mem, int node)
{
	int numa_node = -1;

	if (get_mempolicy(&numa_node, NULL, 0, mem, MPOL_F_NODE | MPOL_F_ADDR) < 0)
		printf("WARNING: get_mempolicy failed");
	assert(numa_node == node);
}

static void *thread_run(void *_arg)
{
	struct thr_param *arg = _arg;
	unsigned long long ns;

	self_setaffinity(arg->cpu);

	atomic_inc(&arg->p->ready);
	while (arg->p->lock)
		barrier();

	ns = nsecs();
	arg->p->burn_fn(arg);
	ns = nsecs() - ns;

	arg->p->ns[arg->cpu] = ns;

	return NULL;
}

static void stat(struct param *p, double *_min, double *_max,
		 double *avg, double *dev)
{
	double sum;
	double min = ULLONG_MAX, max = 0;
	int cpu, cpu_nr;

	cpu_nr = numa_bitmask_weight(p->cpum);

	for (sum = 0, cpu = 0; cpu < p->cpum->size; cpu++) {
		if (!numa_bitmask_isbitset(p->cpum, cpu))
			continue;
		min = p->ns[cpu] < min ? p->ns[cpu] : min;
		max = p->ns[cpu] > max ? p->ns[cpu] : max;
		sum += p->ns[cpu];
	}
	*avg = sum / cpu_nr;

	for (sum = 0, cpu = 0; cpu < p->cpum->size; cpu++) {
		if (!numa_bitmask_isbitset(p->cpum, cpu))
			continue;
		sum += (p->ns[cpu] - *avg) * (p->ns[cpu] - *avg);
	}
	if (cpu_nr > 1)
		*dev = sqrt(sum / (cpu_nr - 1));
	else
		*dev = 0;
	*_min = min;
	*_max = max;
}

static void run_test(int node, struct bitmask *cpum, int dist,
		     burn_fn_t burn_fn,  double *min, double *max,
		     double *avg, double *dev)
{
	int cpu_nr = numa_bitmask_weight(cpum);
	pthread_t threads[cpu_nr];
	struct thr_param params[cpu_nr];
	struct param *p;
	int cpu, i, err;

	assert(cpu_nr > 0);

	p = malloc(sizeof(*p));
	assert(p != NULL);

	memset(p, 0, sizeof(*p));
	p->cpum = cpum;
	p->dist = dist;
	p->lock = 1;
	p->burn_fn = burn_fn;

	numa_set_strict(1);
	p->data = numa_alloc_onnode(sizeof(data_arr_t), node);
	assert(p->data != NULL);
	memset(p->data, 0, sizeof(data_arr_t));

	/* We check the address only after memset, i.e. after
	   first page fault */
	check_memory_node_for_addr(p->data, node);

	for (i = 0, cpu = 0; cpu < cpum->size; cpu++) {
		if (!numa_bitmask_isbitset(cpum, cpu))
			continue;
		params[i].cpu = cpu;
		params[i].p   = p;
		err = pthread_create(&threads[i], NULL,
				     thread_run, &params[i]);
		assert(err == 0);
		i++;
	}
	assert(i == cpu_nr);

	while (atomic_read(&p->ready) != cpu_nr)
		barrier();
	p->lock = 0;

	for (i = 0; i < cpu_nr; i++)
		pthread_join(threads[i], NULL);

	stat(p, min, max, avg, dev);
	numa_free(p->data, sizeof(data_arr_t));
	free(p);
}

static int find_first_setbit(const struct bitmask *bm)
{
	int bit;

	for (bit = 0; bit < bm->size; bit++)
		if (numa_bitmask_isbitset(bm, bit))
			return bit;
	return bit;
}

static int find_last_setbit(const struct bitmask *bm)
{
	int bit;

	for (bit = bm->size - 1; bit >= 0; bit--)
		if (numa_bitmask_isbitset(bm, bit))
			return bit;
	return bm->size;
}

static int find_next_setbit(const struct bitmask *bm, int bit)
{
	for (; bit < bm->size; bit++)
		if (numa_bitmask_isbitset(bm, bit))
			return bit;
	return bm->size;
}

static inline void bitmask_print(const struct bitmask *bm,
				 const char *name)
{
	int bit;

	printf("%15s: ", name);
	for (bit = 0; bit < bm->size; bit++)
		printf("%d", !!numa_bitmask_isbitset(bm, bit));
	printf("\n");
}

static void bitmask_and(struct bitmask *res,
			const struct bitmask *bm1,
			const struct bitmask *bm2)
{
	int bit;

	assert(bm1->size == bm2->size);
	assert(res->size == bm1->size);

	for (bit = 0; bit < res->size; bit++)
		if (numa_bitmask_isbitset(bm1, bit) &&
		    numa_bitmask_isbitset(bm2, bit))
			numa_bitmask_setbit(res, bit);
		else
			numa_bitmask_clearbit(res, bit);
}

static void bitmask_or(struct bitmask *res,
		       const struct bitmask *bm1,
		       const struct bitmask *bm2)
{
	int bit;

	assert(bm1->size == bm2->size);
	assert(res->size == bm1->size);

	for (bit = 0; bit < res->size; bit++)
		if (numa_bitmask_isbitset(bm1, bit) ||
		    numa_bitmask_isbitset(bm2, bit))
			numa_bitmask_setbit(res, bit);
		else
			numa_bitmask_clearbit(res, bit);
}

static int bb_last_setbit(const struct bitmask *mask,
			  const struct bitmask *bm)
{
	int bit;

	assert(mask->size == bm->size);

	for (bit = mask->size - 1; bit >= 0; bit--) {
		if (numa_bitmask_isbitset(mask, bit) &&
		    numa_bitmask_isbitset(bm, bit))
			return bit;
	}
	return mask->size;
}

static int bb_first_zerobit(const struct bitmask *mask,
			    const struct bitmask *bm)
{
	int bit;

	assert(mask->size == bm->size);

	for (bit = 0; bit < mask->size; bit++) {
		if (numa_bitmask_isbitset(mask, bit) &&
		    !numa_bitmask_isbitset(bm, bit))
			return bit;
	}
	return mask->size;
}

static int bb_first_setbit(const struct bitmask *mask,
			   const struct bitmask *bm)
{
	int bit;

	assert(mask->size == bm->size);

	for (bit = 0; bit < mask->size; bit++) {
		if (numa_bitmask_isbitset(mask, bit) &&
		    numa_bitmask_isbitset(bm, bit))
			return bit;
	}
	return mask->size;
}

static int cpus_on_node(const struct bitmask *cpum,
			struct bitmask *cpus_on_node,
			int node)
{
	int err;

	err = numa_node_to_cpus(node, cpus_on_node);
	assert(err == 0);
	bitmask_and(cpus_on_node, cpus_on_node, cpum);
	return numa_bitmask_weight(cpus_on_node);
}

static int nr_cpus_on_node(const struct bitmask *cpum, int node)
{
	int nr;
	struct bitmask *cpus;

	cpus = numa_allocate_cpumask();
	assert(cpus != NULL);
	nr = cpus_on_node(cpum, cpus, node);
	numa_free_cpumask(cpus);
	return nr;
}

static int bb_find_node_with_min_cpus(const struct bitmask *nodm,
				      const struct bitmask *nodes,
				      const struct bitmask *cpus,
				      int beg, int end)
{
	int node, cpus_nr;
	int min_node = nodm->size, min_cpus = INT_MAX;

	assert(nodm->size == nodes->size);
	assert(beg < end);
	assert(beg >= 0);
	assert(end <= nodm->size);

	for (node = beg; node < end; node++) {
		if (numa_bitmask_isbitset(nodm, node) &&
		    numa_bitmask_isbitset(nodes, node)) {
			cpus_nr = nr_cpus_on_node(cpus, node);
			if (cpus_nr < min_cpus) {
				min_cpus = cpus_nr;
				min_node = node;
			}
		}
	}

	return min_node;
}

static int bb_find_node_with_max_cpus(const struct bitmask *nodm,
				      const struct bitmask *nodes,
				      const struct bitmask *cpus,
				      int beg, int end)
{
	int node, cpus_nr;
	int max_node = nodm->size, max_cpus = INT_MIN;

	assert(nodm->size == nodes->size);
	assert(beg < end);
	assert(beg >= 0);
	assert(end <= nodm->size);

	for (node = beg; node < end; node++) {
		if (numa_bitmask_isbitset(nodm, node) &&
		    numa_bitmask_isbitset(nodes, node)) {
			cpus_nr = nr_cpus_on_node(cpus, node);
			if (cpus_nr > max_cpus) {
				max_cpus = cpus_nr;
				max_node = node;
			}
		}
	}

	return max_node;
}

static void b_move_any_cpu_from_to(struct bitmask *cpus,
				   int from_node, int to_node)
{
	int cpu;
	struct bitmask *cpus_on_node;

	cpus_on_node = numa_allocate_cpumask();
	assert(cpus_on_node != NULL);

	numa_node_to_cpus(from_node, cpus_on_node);
	cpu = bb_last_setbit(cpus_on_node, cpus);
	assert(cpu < cpus->size);
	numa_bitmask_clearbit(cpus, cpu);
	numa_bitmask_clearall(cpus_on_node);

	numa_node_to_cpus(to_node, cpus_on_node);
	cpu = bb_first_zerobit(cpus_on_node, cpus);
	assert(cpu < cpus->size);
	numa_bitmask_setbit(cpus, cpu);

	numa_free_cpumask(cpus_on_node);
}

static int intersect_cpumask_and_nodemask(struct bitmask *cpum,
					  struct bitmask *nodm)
{
	struct bitmask *cpus, *cpus_on_node;
	int bit, err;

	cpus_on_node = numa_allocate_cpumask();
	assert(cpus_on_node != NULL);
	cpus = numa_allocate_cpumask();
	assert(cpus != NULL);

	for (bit = 0; bit < nodm->size; bit++) {
		if (!numa_bitmask_isbitset(nodm, bit))
			continue;
		err = numa_node_to_cpus(bit, cpus_on_node);
		assert(err == 0);
		bitmask_and(cpus_on_node, cpus_on_node, cpum);
		if (numa_bitmask_weight(cpus_on_node)) {
			bitmask_or(cpus, cpus, cpus_on_node);
			numa_bitmask_clearall(cpus_on_node);
		}
		else
			numa_bitmask_clearbit(nodm, bit);
	}
	copy_bitmask_to_bitmask(cpus, cpum);

	numa_free_cpumask(cpus_on_node);
	numa_free_cpumask(cpus);

	return numa_bitmask_weight(cpum) && numa_bitmask_weight(nodm);
}

enum {
	EXPAND,
	SHRINK,
};

static int traverse_direction(const struct bitmask *nodm,
			      const struct bitmask *nodes,
			      const struct bitmask *cpus)
{
	int nr_cpus, nr_nodes, nr_on_node;
	int first_node, last_node;

	last_node = find_last_setbit(nodm);
	assert(last_node < nodm->size);
	if (!numa_bitmask_isbitset(nodes, last_node))
		/* We still expand if last node is not yet reached */
		return EXPAND;

	first_node = find_first_setbit(nodm);
	assert(first_node < nodm->size);

	if (!numa_bitmask_isbitset(nodes, first_node))
		/* We definitely shrink if first node is already free */
		return SHRINK;

	nr_nodes = numa_bitmask_weight(nodm);
	nr_cpus = numa_bitmask_weight(cpus);
	nr_on_node = nr_cpus_on_node(cpus, last_node);

	/* We start shrinking if we have enough CPUs on latest node */
	return nr_on_node >= nr_cpus / nr_nodes ? SHRINK : EXPAND;
}

static int do_expand(const struct bitmask *cpum,
		     const struct bitmask *nodm,
		     struct bitmask *cpus,
		     struct bitmask *nodes)
{
	int to_node, from_node;
	int nr, nr_nodes, nr_cpus, nr_on_node;

	nr_nodes = numa_bitmask_weight(nodes);
	nr_cpus  = numa_bitmask_weight(cpus);

	to_node = bb_last_setbit(nodm, nodes);
	assert(to_node < nodm->size);

	nr_on_node = nr_cpus_on_node(cpus, to_node);
	nr = nr_cpus / nr_nodes;
	if (nr == nr_on_node) {
		to_node = bb_first_zerobit(nodm, nodes);
		assert(to_node < nodm->size);
		numa_bitmask_setbit(nodes, to_node);
		nr = nr_cpus / (nr_nodes + 1);
	} else
		assert(nr_on_node < nr);

	from_node = bb_find_node_with_max_cpus(nodm, nodes, cpus,
					       0, to_node);
	assert(from_node < nodm->size);

	b_move_any_cpu_from_to(cpus, from_node, to_node);

	/* Always continue, shrink is left */
	return 1;
}

static int do_shrink(const struct bitmask *cpum,
		     const struct bitmask *nodm,
		     struct bitmask *cpus,
		     struct bitmask *nodes)
{
	int last_node, to_node, from_node;
	int nr_cpus, nr_on_node;

	nr_cpus  = numa_bitmask_weight(cpus);

	last_node = find_last_setbit(nodm);
	assert(last_node < nodm->size);

	if (numa_bitmask_isbitset(nodes, last_node) &&
	    nr_cpus_on_node(cpus, last_node) == nr_cpus) {
		numa_bitmask_clearbit(nodes, last_node);
		/* The End */
		return 0;
	}

	from_node = bb_first_setbit(nodm, nodes);
	assert(from_node < nodm->size);

	nr_on_node = nr_cpus_on_node(cpus, from_node);
	if (nr_on_node == 0) {
		numa_bitmask_clearbit(nodes, from_node);
		from_node = bb_first_setbit(nodm, nodes);
		assert(from_node < nodm->size);
	}

	to_node = bb_find_node_with_min_cpus(nodm, nodes, cpus,
					     from_node + 1,
					     last_node + 1);
	assert(to_node < nodm->size);

	b_move_any_cpu_from_to(cpus, from_node, to_node);

	/* Still going on */
	return 1;
}

static struct bitmask *cpumask__one_iteration(const struct bitmask *cpum,
					     const struct bitmask *nodm,
					     struct bitmask *cpus,
					     struct bitmask **nodes)

{
	/*
	 * Fully copy @cpum to @cpus, only 1 iteration.
	 */
	(void)nodm;
	(void)nodes;

	if (cpus == NULL) {
		cpus = numa_allocate_cpumask();
		assert(cpus != NULL);
		/* XXX Stupid NUMA API, where is const? */
		copy_bitmask_to_bitmask((struct bitmask *)cpum, cpus);
	} else {
		numa_free_cpumask(cpus);
		cpus = NULL;
	}

	return cpus;
}

static struct bitmask *cpumask__cpu_increase(const struct bitmask *cpum,
					     const struct bitmask *nodm,
					     struct bitmask *cpus,
					     struct bitmask **nodes)
{
	/*
	 * Pick up next CPU bit from @cpum and set it in @cpus
	 * one by one on each iteration.
	 */
	int bit;

	if (cpus == NULL) {
		cpus = numa_allocate_cpumask();
		assert(cpus != NULL);
		bit = find_next_setbit(cpum, 0);
		assert(bit < cpum->size);
		numa_bitmask_setbit(cpus, bit);
	} else {
		if (numa_bitmask_equal(cpum, cpus)) {
			numa_free_cpumask(cpus);
			cpus = NULL;
		} else {
			bit = find_last_setbit(cpus);
			bit += 1;
			assert(bit < cpus->size);
			bit = find_next_setbit(cpum, bit);
			assert(bit < cpum->size);
			numa_bitmask_setbit(cpus, bit);
		}
	}

	return cpus;
}

static struct bitmask *cpumask__cpu_rollover(const struct bitmask *cpum,
					     const struct bitmask *nodm,
					     struct bitmask *cpus,
					     struct bitmask **nodes)
{
	int bit, dir, cont;

	if (cpus == NULL) {
		*nodes = numa_allocate_cpumask();
		assert(*nodes != NULL);
		cpus = numa_allocate_cpumask();
		assert(cpus != NULL);
		bit = find_first_setbit(nodm);
		assert(bit < nodm->size);
		cpus_on_node(cpum, cpus, bit);
		assert(numa_bitmask_weight(cpus) > 0);
		numa_bitmask_setbit(*nodes, bit);
	} else {
		dir = traverse_direction(nodm, *nodes, cpus);
		if (dir == EXPAND)
			cont = do_expand(cpum, nodm, cpus, *nodes);
		else
			cont = do_shrink(cpum, nodm, cpus, *nodes);

		if (!cont) {
			numa_free_cpumask(cpus);
			numa_free_cpumask(*nodes);
			cpus = NULL;
		}
	}

	return cpus;
}

static struct bitmask *cpumask__node_cascade(const struct bitmask *cpum,
					     const struct bitmask *nodm,
					     struct bitmask *cpus,
					     struct bitmask **nodes)
{
	int bit;

	if (cpus == NULL) {
		*nodes = numa_allocate_cpumask();
		assert(*nodes != NULL);
		cpus = numa_allocate_cpumask();
		assert(cpus != NULL);
		bit = find_first_setbit(nodm);
		assert(bit < nodm->size);
		numa_bitmask_setbit(*nodes, bit);
		cpus_on_node(cpum, cpus, bit);
	} else {
		bit = find_last_setbit(*nodes);
		assert(bit < (*nodes)->size);
		numa_bitmask_clearbit(*nodes, bit);
		bit = find_next_setbit(nodm, bit + 1);
		if (bit >= nodm->size) {
			numa_free_cpumask(cpus);
			numa_free_cpumask(*nodes);
			cpus = NULL;
		} else {
			numa_bitmask_setbit(*nodes, bit);
			numa_bitmask_clearall(cpus);
			cpus_on_node(cpum, cpus, bit);
		}
	}

	return cpus;
}

typedef struct bitmask *(* cpumask_fn_t)(const struct bitmask *cpum,
					 const struct bitmask *nodm,
					 struct bitmask *cpus,
					 struct bitmask **nodes);

struct load_ent {
	const char    *name;
	const char    *desc;
	cpumask_fn_t  cpumask;
} const load_table[] = {
	{"one-iteration",   "Burn everything we can.",
	 cpumask__one_iteration},
	{"cpu-increase",    "Increase CPUs amount one by one.",
	 cpumask__cpu_increase },
	{"cpu-rollover",    "Rollover on CPUs from first node to last node.",
	 cpumask__cpu_rollover },
	{"node-cascade",    "Burn all CPUs on each node.",
	 cpumask__node_cascade},
};

static struct bitmask *cpumask_for_test(const struct load_ent *load_ent,
					const struct bitmask *cpum,
					const struct bitmask *nodm,
					struct bitmask *cpus,
					struct bitmask **nodes)
{
	return load_ent->cpumask(cpum, nodm, cpus, nodes);
}

static void print_test_header(const char *host,
			      const struct bitmask *cpus,
			      const struct bitmask *nodm)
{
	struct bitmask *cpus_on_node;
	int err, len, node, nr_nodes, nr_on_node;
	char buf[16];

	cpus_on_node = numa_allocate_cpumask();
	assert(cpus_on_node != NULL);
	nr_nodes = numa_num_configured_nodes();

	printf("Nodes ");
	for (node = 0; node < nr_nodes; node++) {
		err = numa_node_to_cpus(node, cpus_on_node);
		assert(err == 0);
		nr_on_node = numa_bitmask_weight(cpus_on_node);
		numa_bitmask_clearall(cpus_on_node);
		len = snprintf(buf, sizeof(buf), "N%d", node);
		if (len < nr_on_node) {
			len = nr_on_node/2 - len/2 + len;
			printf("%*s%*s", len, buf, nr_on_node - len, " ");
		} else
			/* Nodes number bigger than amount of cpus? Come on */
			assert(0);
		printf(" ");
	}
	numa_free_cpumask(cpus_on_node);
	printf("CPUs %2s operation %5s min %5s max %5s avg %3s stdev\n",
	       "", "", "", "", "");
}

static void print_test_result(const struct burn_ent *burn_ent,
			      const struct bitmask *cpus,
			      const struct bitmask *nodm,
			      double min, double max,
			      double avg, double dev)
{
	int err, node, nr_nodes, cpu;
	int first_cpu, last_cpu;
	struct bitmask *cpus_on_node;

	cpus_on_node = numa_allocate_cpumask();
	assert(cpus_on_node != NULL);
	nr_nodes = numa_num_configured_nodes();

	printf(" CPUs ");
	for (node = 0; node < nr_nodes; node++) {
		err = numa_node_to_cpus(node, cpus_on_node);
		assert(err == 0);
		first_cpu = find_first_setbit(cpus_on_node);
		assert(first_cpu < cpus_on_node->size);
		last_cpu = find_last_setbit(cpus_on_node);
		assert(last_cpu < cpus_on_node->size);
		numa_bitmask_clearall(cpus_on_node);

		for (cpu = first_cpu; cpu <= last_cpu; cpu++)
			printf("%c", numa_bitmask_isbitset(cpus, cpu) ?
			       '*' : '-');
		printf(" ");
	}
	numa_free_cpumask(cpus_on_node);

	printf("%4d %12s %9.3f %9.3f %9.3f %9.3f\n",
	       numa_bitmask_weight(cpus),
	       burn_ent->name,
	       min/SPINS,
	       max/SPINS,
	       avg/SPINS,
	       dev/SPINS);
}

static void print_test_footer(void)
{
	printf("\n");
}

static void usage(void)
{
	char burn_buf[4096];
	char load_buf[4096];
	int i, sz = 0;

	for (i = 0; i < ARRAY_LENGTH(burn_table); i++) {
		const struct burn_ent *e = &burn_table[i];

		sz += snprintf(burn_buf + sz, sizeof(burn_buf) - sz,
			       "                   %10s  %s\n",
			       e->name,
			       e->desc);
	}
	for (sz = 0, i = 1; i < ARRAY_LENGTH(load_table); i++) {
		const struct load_ent *e = &load_table[i];

		sz += snprintf(load_buf + sz, sizeof(load_buf) - sz,
			       "         %20s  %s\n",
			       e->name,
			       e->desc);
	}
	printf("Cache contention measurement tool.\n\n"
	       "\t-l|--load  <string>    Load pattern:\n\n"
	       "%s\n"
	       "\t-o|--op    <name>      Operation name:\n\n"
	       "%s\n"
	       "\t-c|--cpu   <string>    String of CPUs to burn, e.g.:\n"
	       "\t                          1-5,7,10  or 4-5.\n"
	       "\t                       By default all CPUS are burned.\n"
	       "\t-n|--nodes <string>    Node nodes to use, e.g.:\n"
	       "\t                          1-5,7,10  or 4-5.\n"
	       "\t                       By default all nodes are used.\n"
	       "\t-d|--dist <number>     Distance between data for each CPU, 0 by default.\n"
	       "\t                       Value should be between 0 and %d, for bit operations \n"
	       "\t                       this value represents power of two, i.e.\n"
	       "\t                          bit = current_cpu << dist;\n"
	       "\t                       for other operations is always a multiplier, i.e. \n"
	       "\t                          ind = current_cpu * dist;\n"
	       "\t-h|--help              Show usage information.\n"
	       "\n",
	       load_buf,
	       burn_buf,
	       DIST_MAX);
}

struct option opts[] = {
	{ "load",  required_argument, NULL, 'l' },
	{ "op" ,   required_argument, NULL, 'o' },
	{ "cpu",   required_argument, NULL, 'c' },
	{ "numa",  required_argument, NULL, 'n' },
	{ "dist",  required_argument, NULL, 'd' },
	{ "help",  no_argument,       NULL, 'h' },
	{ NULL, 0, NULL, 0 }
};
const char *opts_str = "l:o:c:n:d:h";

int main(int argc, char *argv[])
{
	double min, max, avg, dev;
	struct bitmask *nodm = NULL;
	struct bitmask *cpum = NULL;
	const struct burn_ent *burn_ent = NULL, *burn_end;
	const struct load_ent *load_ent = &load_table[0];
	char host[HOST_NAME_MAX];
	int dist = 0;
	int i, c;

	if (numa_num_possible_cpus() > CPU_MAX) {
		printf("Too many CPUs on this machine. Change CPU_MAX macro.\n");
		exit(1);
	}
	if (numa_available()) {
		printf("NUMA is not available on this machine.\n");
		exit(1);
	}

	/* Always CPU#0 */
	self_setaffinity(0);

	while ((c = getopt_long(argc, argv, opts_str,
				opts, NULL)) != -1) {
		switch (c) {
		case 'h':
			usage();
			return 0;
		case 'l':
			load_ent = NULL;
			for (i = 0; i < ARRAY_LENGTH(load_table); i++) {
				const struct load_ent *e = &load_table[i];

				if (0 == strcmp(optarg, e->name)) {
					load_ent = e;
					break;
				}
			}
			if (load_ent == NULL) {
				printf("Load pattern is incorrect, see usage.\n");
				exit(1);
			}
			break;
		case 'o':
			for (i = 0; i < ARRAY_LENGTH(burn_table); i++) {
				const struct burn_ent *e = &burn_table[i];

				if (0 == strcmp(optarg, e->name)) {
						burn_ent = e;
						break;
				}
			}
			if (burn_ent == NULL) {
				printf("Operation is incorrect, see usage.\n");
				exit(1);
			}
			break;
		case 'c':
			cpum = numa_parse_cpustring_all(optarg);
			if (cpum == NULL) {
				printf("Can't parse CPU string '%s'\n", optarg);
				exit(1);
			}
			break;
		case 'n':
			nodm = numa_parse_nodestring_all(optarg);
			if (nodm == NULL) {
				printf("Can't parse NUMA nodes string '%s'\n", optarg);
				exit(1);
			}
			break;
		case 'd':
			dist = atoi(optarg);
			if (dist < 0 || dist > DIST_MAX) {
				printf("Distance is incorrect.\n");
				exit(1);
			}
			break;
		default:
			return 1;
		}
	}

	if (cpum == NULL) {
		cpum = numa_allocate_cpumask();
		assert(cpum != NULL);
		copy_bitmask_to_bitmask(numa_all_cpus_ptr, cpum);
	}
	if (nodm == NULL) {
		nodm = numa_allocate_cpumask();
		assert(nodm != NULL);
		copy_bitmask_to_bitmask(numa_all_nodes_ptr, nodm);
	}
	if (!intersect_cpumask_and_nodemask(cpum, nodm)) {
		printf("No CPUs or NUMA nodes.\n");
		exit(1);
	}
	if (burn_ent == NULL) {
		burn_ent = &burn_table[0];
		burn_end = burn_ent + ARRAY_LENGTH(burn_table);
	} else
		burn_end = burn_ent + 1;

	gethostname(host, sizeof(host));

	for (; burn_ent < burn_end; burn_ent++) {
		struct bitmask *cpus = NULL, *nodes = NULL;

		print_test_header(host, cpum, nodm);

		while ((cpus = cpumask_for_test(load_ent, cpum, nodm,
						cpus, &nodes))) {
			run_test(numa_node_of_cpu(0), cpus, dist,
				 burn_ent->fn, &min, &max, &avg, &dev);
			print_test_result(burn_ent, cpus, nodm,
					  min, max, avg, dev);
		}
		print_test_footer();
	}

	numa_free_cpumask(cpum);
	numa_free_nodemask(nodm);

	return 0;
}

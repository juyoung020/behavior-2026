// ===== 원본: physx/include/foundation/PxHashInternals.h
namespace px
{
template <class Entry, class Key, class HashFn, class GetKey, class PxAllocator, bool compacting>
class PxHashBase : private PxAllocator
{
	void init(uint32_t initialTableSize, float loadFactor)
	{
		mBuffer = NULL;
		mEntries = NULL;
		mEntriesNext = NULL;
		mHash = NULL;
		mEntriesCapacity = 0;
		mHashSize = 0;
		mLoadFactor = loadFactor;
		mFreeList = uint32_t(EOL);
		mTimestamp = 0;
		mEntriesCount = 0;

		if(initialTableSize)
			reserveInternal(initialTableSize);
	}

  public:
	typedef Entry EntryType;

	PxHashBase(uint32_t initialTableSize = 64, float loadFactor = 0.75f) : PxAllocator("hashBase")
	{
		init(initialTableSize, loadFactor);
	}

	PxHashBase(uint32_t initialTableSize, float loadFactor, const PxAllocator& alloc) : PxAllocator(alloc)
	{
		init(initialTableSize, loadFactor);
	}

	PxHashBase(const PxAllocator& alloc) : PxAllocator(alloc)
	{
		init(64, 0.75f);
	}

	~PxHashBase()
	{
		destroy(); // No need to clear()

		if(mBuffer)
			PxAllocator::deallocate(mBuffer);
	}

	static const uint32_t EOL = 0xffffffff;

	EHDI Entry* create(const Key& k, bool& exists)
	{
		uint32_t h = 0;
		if(mHashSize)
		{
			h = hash(k);
			uint32_t index = mHash[h];
			while(index != EOL && !HashFn().equal(GetKey()(mEntries[index]), k))
				index = mEntriesNext[index];
			exists = index != EOL;
			if(exists)
				return mEntries + index;
		}
		else
			exists = false;

		if(freeListEmpty())
		{
			grow();
			h = hash(k);
		}

		uint32_t entryIndex = freeListGetNext();

		mEntriesNext[entryIndex] = mHash[h];
		mHash[h] = entryIndex;

		mEntriesCount++;
		mTimestamp++;

		return mEntries + entryIndex;
	}

	EHDI const Entry* find(const Key& k) const
	{
		if(!mEntriesCount)
			return NULL;

		const uint32_t h = hash(k);
		uint32_t index = mHash[h];
		while(index != EOL && !HashFn().equal(GetKey()(mEntries[index]), k))
			index = mEntriesNext[index];
		return index != EOL ? mEntries + index : NULL;
	}

	EHDI bool erase(const Key& k, Entry& e)
	{
		if(!mEntriesCount)
			return false;

		const uint32_t h = hash(k);
		uint32_t* ptr = mHash + h;
		while(*ptr != EOL && !HashFn().equal(GetKey()(mEntries[*ptr]), k))
			ptr = mEntriesNext + *ptr;

		if(*ptr == EOL)
			return false;

		E_PLACEMENT_NEW(&e, Entry)(mEntries[*ptr]);		

		return eraseInternal(ptr);
	}

	EHDI bool erase(const Key& k)
	{
		if(!mEntriesCount)
			return false;

		const uint32_t h = hash(k);
		uint32_t* ptr = mHash + h;
		while(*ptr != EOL && !HashFn().equal(GetKey()(mEntries[*ptr]), k))
			ptr = mEntriesNext + *ptr;

		if(*ptr == EOL)
			return false;		

		return eraseInternal(ptr);
	}

	EHDI uint32_t size() const
	{
		return mEntriesCount;
	}

	EHDI uint32_t capacity() const
	{
		return mHashSize;
	}

	void clear()
	{
		if(!mHashSize || mEntriesCount == 0)
			return;

		destroy();

		intrinsics::memSet(mHash, EOL, mHashSize * sizeof(uint32_t));

		const uint32_t sizeMinus1 = mEntriesCapacity - 1;
		for(uint32_t i = 0; i < sizeMinus1; i++)
		{
			PxPrefetchLine(mEntriesNext + i, 128);
			mEntriesNext[i] = i + 1;
		}
		mEntriesNext[mEntriesCapacity - 1] = uint32_t(EOL);
		mFreeList = 0;
		mEntriesCount = 0;
	}

	void reserve(uint32_t size)
	{
		if(size > mHashSize)
			reserveInternal(size);
	}

	EHDI const Entry* getEntries() const
	{
		return mEntries;
	}

	EHDI Entry* insertUnique(const Key& k)
	{
		E_ASSERT(find(k) == NULL);
		uint32_t h = hash(k);

		uint32_t entryIndex = freeListGetNext();

		mEntriesNext[entryIndex] = mHash[h];
		mHash[h] = entryIndex;

		mEntriesCount++;
		mTimestamp++;

		return mEntries + entryIndex;
	}

  private:
	void destroy()
	{
		for(uint32_t i = 0; i < mHashSize; i++)
		{
			for(uint32_t j = mHash[i]; j != EOL; j = mEntriesNext[j])
				mEntries[j].~Entry();
		}
	}

	template <typename HK, typename GK, class A, bool comp>
	 void copy(const PxHashBase<Entry, Key, HK, GK, A, comp>& other);

	// free list management - if we're coalescing, then we use mFreeList to hold
	// the top of the free list and it should always be equal to size(). Otherwise,
	// we build a free list in the next() pointers.

	EHDI void freeListAdd(uint32_t index)
	{
		if(compacting)
		{
			mFreeList--;
			E_ASSERT(mFreeList == mEntriesCount);
		}
		else
		{
			mEntriesNext[index] = mFreeList;
			mFreeList = index;
		}
	}

	EHDI void freeListAdd(uint32_t start, uint32_t end)
	{
		if(!compacting)
		{
			for(uint32_t i = start; i < end - 1; i++) // add the new entries to the free list
				mEntriesNext[i] = i + 1;

			// link in old free list
			mEntriesNext[end - 1] = mFreeList;
			E_ASSERT(mFreeList != end - 1);
			mFreeList = start;
		}
		else if(mFreeList == EOL) // don't reset the free ptr for the compacting hash unless it's empty
			mFreeList = start;
	}

	EHDI uint32_t freeListGetNext()
	{
		E_ASSERT(!freeListEmpty());
		if(compacting)
		{
			E_ASSERT(mFreeList == mEntriesCount);
			return mFreeList++;
		}
		else
		{
			uint32_t entryIndex = mFreeList;
			mFreeList = mEntriesNext[mFreeList];
			return entryIndex;
		}
	}

	EHDI bool freeListEmpty() const
	{
		if(compacting)
			return mEntriesCount == mEntriesCapacity;
		else
			return mFreeList == EOL;
	}

	EHDI void replaceWithLast(uint32_t index)
	{
		E_PLACEMENT_NEW(mEntries + index, Entry)(mEntries[mEntriesCount]);
		mEntries[mEntriesCount].~Entry();
		mEntriesNext[index] = mEntriesNext[mEntriesCount];

		uint32_t h = hash(GetKey()(mEntries[index]));
		uint32_t* ptr;
		for(ptr = mHash + h; *ptr != mEntriesCount; ptr = mEntriesNext + *ptr)
			E_ASSERT(*ptr != EOL);
		*ptr = index;
	}

	EHDI uint32_t hash(const Key& k, uint32_t hashSize) const
	{
		return HashFn()(k) & (hashSize - 1);
	}

	EHDI uint32_t hash(const Key& k) const
	{
		return hash(k, mHashSize);
	}

	EHDI bool eraseInternal(uint32_t* ptr)
	{
		const uint32_t index = *ptr;

		*ptr = mEntriesNext[index];

		mEntries[index].~Entry();

		mEntriesCount--;
		mTimestamp++;

		if (compacting && index != mEntriesCount)
			replaceWithLast(index);

		freeListAdd(index);
		return true;
	}

	 void reserveInternal(uint32_t size)
	{
		if(!PxIsPowerOfTwo(size))
			size = PxNextPowerOfTwo(size);

		E_ASSERT(!(size & (size - 1)));

		// decide whether iteration can be done on the entries directly
		bool resizeCompact = compacting || freeListEmpty();

		// define new table sizes
		uint32_t oldEntriesCapacity = mEntriesCapacity;
		uint32_t newEntriesCapacity = uint32_t(float(size) * mLoadFactor);
		uint32_t newHashSize = size;

		// allocate new common buffer and setup pointers to new tables
		uint8_t* newBuffer;
		uint32_t* newHash;
		uint32_t* newEntriesNext;
		Entry* newEntries;
		{
			const uint64_t newEntriesNextBytesOffset = newHashSize * sizeof(uint32_t);
			uint64_t newEntriesByteOffset = newEntriesNextBytesOffset + newEntriesCapacity * sizeof(uint32_t);
			newEntriesByteOffset += (16 - (newEntriesByteOffset & 15)) & 15;
			const uint64_t newBufferByteSize = newEntriesByteOffset + newEntriesCapacity * sizeof(Entry);

			newBuffer = reinterpret_cast<uint8_t*>(PxAllocator::allocate(newBufferByteSize, EPX_FL));
			E_ASSERT(newBuffer);

			newHash = reinterpret_cast<uint32_t*>(newBuffer);
			newEntriesNext = reinterpret_cast<uint32_t*>(newBuffer + newEntriesNextBytesOffset);
			newEntries = reinterpret_cast<Entry*>(newBuffer + newEntriesByteOffset);
		}

		// initialize new hash table
		intrinsics::memSet(newHash, int32_t(EOL), newHashSize * sizeof(uint32_t));

		// iterate over old entries, re-hash and create new entries
		if(resizeCompact)
		{
			// check that old free list is empty - we don't need to copy the next entries
			E_ASSERT(compacting || mFreeList == EOL);

			for(uint32_t index = 0; index < mEntriesCount; ++index)
			{
				uint32_t h = hash(GetKey()(mEntries[index]), newHashSize);
				newEntriesNext[index] = newHash[h];
				newHash[h] = index;

				E_PLACEMENT_NEW(newEntries + index, Entry)(mEntries[index]);
				mEntries[index].~Entry();
			}
		}
		else
		{
			// copy old free list, only required for non compact resizing
			intrinsics::memCopy(newEntriesNext, mEntriesNext, mEntriesCapacity * sizeof(uint32_t));

			for(uint32_t bucket = 0; bucket < mHashSize; bucket++)
			{
				uint32_t index = mHash[bucket];
				while(index != EOL)
				{
					uint32_t h = hash(GetKey()(mEntries[index]), newHashSize);
					newEntriesNext[index] = newHash[h];
					E_ASSERT(index != newHash[h]);

					newHash[h] = index;

					E_PLACEMENT_NEW(newEntries + index, Entry)(mEntries[index]);
					mEntries[index].~Entry();

					index = mEntriesNext[index];
				}
			}
		}

		// swap buffer and pointers
		PxAllocator::deallocate(mBuffer);
		mBuffer = newBuffer;
		mHash = newHash;
		mHashSize = newHashSize;
		mEntriesNext = newEntriesNext;
		mEntries = newEntries;
		mEntriesCapacity = newEntriesCapacity;

		freeListAdd(oldEntriesCapacity, newEntriesCapacity);
	}

	void grow()
	{
		E_ASSERT((mFreeList == EOL) || (compacting && (mEntriesCount == mEntriesCapacity)));

		uint32_t size = mHashSize == 0 ? 16 : mHashSize * 2;
		reserve(size);
	}

	uint8_t* mBuffer;
	Entry* mEntries;
	uint32_t* mEntriesNext; // same size as mEntries
	uint32_t* mHash;
	uint32_t mEntriesCapacity;
	uint32_t mHashSize;
	float mLoadFactor;
	uint32_t mFreeList;
	uint32_t mTimestamp;
	uint32_t mEntriesCount; // number of entries

  public:
	class Iter
	{
	  public:
		EHDI Iter(PxHashBase& b) : mBucket(0), mEntry(uint32_t(b.EOL)), mTimestamp(b.mTimestamp), mBase(b)
		{
			if(mBase.mEntriesCapacity > 0)
			{
				mEntry = mBase.mHash[0];
				skip();
			}
		}

		EHDI void check() const
		{
			E_ASSERT(mTimestamp == mBase.mTimestamp);
		}
		EHDI const Entry& operator*() const
		{
			check();
			return mBase.mEntries[mEntry];
		}
		EHDI Entry& operator*()
		{
			check();
			return mBase.mEntries[mEntry];
		}
		EHDI const Entry* operator->() const
		{
			check();
			return mBase.mEntries + mEntry;
		}
		EHDI Entry* operator->()
		{
			check();
			return mBase.mEntries + mEntry;
		}
		EHDI Iter operator++()
		{
			check();
			advance();
			return *this;
		}
		EHDI Iter operator++(int)
		{
			check();
			Iter i = *this;
			advance();
			return i;
		}
		EHDI bool done() const
		{
			check();
			return mEntry == mBase.EOL;
		}

	  private:
		EHDI void advance()
		{
			mEntry = mBase.mEntriesNext[mEntry];
			skip();
		}
		EHDI void skip()
		{
			while(mEntry == mBase.EOL)
			{
				if(++mBucket == mBase.mHashSize)
					break;
				mEntry = mBase.mHash[mBucket];
			}
		}

		Iter& operator=(const Iter&);

		uint32_t mBucket;
		uint32_t mEntry;
		uint32_t mTimestamp;
		PxHashBase& mBase;
	};

	/*!
	Iterate over entries in a hash base and allow entry erase while iterating
	*/
	class PxEraseIterator
	{
	public:
		EHDI PxEraseIterator(PxHashBase& b): mBase(b)
		{
			reset();
		}

		EHDI Entry* eraseCurrentGetNext(bool eraseCurrent)
		{
			if(eraseCurrent && mCurrentEntryIndexPtr)
			{
				mBase.eraseInternal(mCurrentEntryIndexPtr);
				// if next was valid return the same ptr, if next was EOL search new hash entry
				if(*mCurrentEntryIndexPtr != mBase.EOL)
					return mBase.mEntries + *mCurrentEntryIndexPtr;
				else
					return traverseHashEntries();
			}

			// traverse mHash to find next entry
			if(mCurrentEntryIndexPtr == NULL)
				return traverseHashEntries();
			
			const uint32_t index = *mCurrentEntryIndexPtr;			
			if(mBase.mEntriesNext[index] == mBase.EOL)
			{
				return traverseHashEntries();
			}
			else
			{
				mCurrentEntryIndexPtr = mBase.mEntriesNext + index;
				return mBase.mEntries + *mCurrentEntryIndexPtr;
			}
		}

		EHDI void reset()
		{
			mCurrentHashIndex = 0;
			mCurrentEntryIndexPtr = NULL;			
		}

	private:
		EHDI Entry* traverseHashEntries()
		{
			mCurrentEntryIndexPtr = NULL;			
			while (mCurrentEntryIndexPtr == NULL && mCurrentHashIndex < mBase.mHashSize)
			{
				if (mBase.mHash[mCurrentHashIndex] != mBase.EOL)
				{
					mCurrentEntryIndexPtr = mBase.mHash + mCurrentHashIndex;
					mCurrentHashIndex++;
					return mBase.mEntries + *mCurrentEntryIndexPtr;
				}
				else
				{
					mCurrentHashIndex++;
				}
			}
			return NULL;
		}

		PxEraseIterator& operator=(const PxEraseIterator&);
	private:
		uint32_t*	mCurrentEntryIndexPtr;
		uint32_t	mCurrentHashIndex;		
		PxHashBase&	mBase;
	};
};

template <class Entry, class Key, class HashFn, class GetKey, class PxAllocator, bool compacting>
template <typename HK, typename GK, class A, bool comp>
 void
PxHashBase<Entry, Key, HashFn, GetKey, PxAllocator, compacting>::copy(const PxHashBase<Entry, Key, HK, GK, A, comp>& other)
{
	reserve(other.mEntriesCount);

	for(uint32_t i = 0; i < other.mEntriesCount; i++)
	{
		for(uint32_t j = other.mHash[i]; j != EOL; j = other.mEntriesNext[j])
		{
			const Entry& otherEntry = other.mEntries[j];

			bool exists;
			Entry* newEntry = create(GK()(otherEntry), exists);
			E_ASSERT(!exists);

			E_PLACEMENT_NEW(newEntry, Entry)(otherEntry);
		}
	}
}

template <class Key, class HashFn, class PxAllocator = typename PxAllocatorTraits<Key>::Type, bool Coalesced = false>
class PxHashSetBase
{
	E_NOCOPY(PxHashSetBase)
  public:
	struct GetKey
	{
		EHDI const Key& operator()(const Key& e)
		{
			return e;
		}
	};

	typedef PxHashBase<Key, Key, HashFn, GetKey, PxAllocator, Coalesced> BaseMap;
	typedef typename BaseMap::Iter Iterator;

	PxHashSetBase(uint32_t initialTableSize, float loadFactor, const PxAllocator& alloc)
	: mBase(initialTableSize, loadFactor, alloc)
	{
	}

	PxHashSetBase(const PxAllocator& alloc) : mBase(64, 0.75f, alloc)
	{
	}

	PxHashSetBase(uint32_t initialTableSize = 64, float loadFactor = 0.75f) : mBase(initialTableSize, loadFactor)
	{
	}

	bool insert(const Key& k)
	{
		bool exists;
		Key* e = mBase.create(k, exists);
		if(!exists)
			E_PLACEMENT_NEW(e, Key)(k);
		return !exists;
	}

	EHDI bool contains(const Key& k) const
	{
		return mBase.find(k) != 0;
	}
	EHDI bool erase(const Key& k)
	{
		return mBase.erase(k);
	}
	EHDI uint32_t size() const
	{
		return mBase.size();
	}
	EHDI uint32_t capacity() const
	{
		return mBase.capacity();
	}
	EHDI void reserve(uint32_t size)
	{
		mBase.reserve(size);
	}
	EHDI void clear()
	{
		mBase.clear();
	}

  protected:
	BaseMap mBase;
};

template <class Key, class Value, class HashFn, class PxAllocator = typename PxAllocatorTraits<PxPair<const Key, Value> >::Type>
class PxHashMapBase
{
	E_NOCOPY(PxHashMapBase)
  public:
	typedef PxPair<const Key, Value> Entry;

	struct GetKey
	{
		EHDI const Key& operator()(const Entry& e)
		{
			return e.first;
		}
	};

	typedef PxHashBase<Entry, Key, HashFn, GetKey, PxAllocator, true> BaseMap;
	typedef typename BaseMap::Iter Iterator;
	typedef typename BaseMap::PxEraseIterator EraseIterator;

	PxHashMapBase(uint32_t initialTableSize, float loadFactor, const PxAllocator& alloc)
	: mBase(initialTableSize, loadFactor, alloc)
	{
	}

	PxHashMapBase(const PxAllocator& alloc) : mBase(64, 0.75f, alloc)
	{
	}

	PxHashMapBase(uint32_t initialTableSize = 64, float loadFactor = 0.75f) : mBase(initialTableSize, loadFactor)
	{
	}

	bool insert(const Key /*&*/ k, const Value /*&*/ v)
	{
		bool exists;
		Entry* e = mBase.create(k, exists);
		if(!exists)
			E_PLACEMENT_NEW(e, Entry)(k, v);
		return !exists;
	}

	Value& operator[](const Key& k)
	{
		bool exists;
		Entry* e = mBase.create(k, exists);
		if(!exists)
			E_PLACEMENT_NEW(e, Entry)(k, Value());

		return e->second;
	}

	EHDI const Entry* find(const Key& k) const
	{
		return mBase.find(k);
	}
	EHDI bool erase(const Key& k)
	{
		return mBase.erase(k);
	}
	EHDI bool erase(const Key& k, Entry& e)
	{		
		return mBase.erase(k, e);
	}
	EHDI uint32_t size() const
	{
		return mBase.size();
	}
	EHDI uint32_t capacity() const
	{
		return mBase.capacity();
	}
	EHDI Iterator getIterator()
	{
		return Iterator(mBase);
	}
	EHDI EraseIterator getEraseIterator()
	{
		return EraseIterator(mBase);
	}
	EHDI void reserve(uint32_t size)
	{
		mBase.reserve(size);
	}
	EHDI void clear()
	{
		mBase.clear();
	}

  protected:
	BaseMap mBase;
};
} // namespace px

// ===== 원본: physx/include/foundation/PxHashMap.h
// TODO: make this doxy-format
//
// This header defines two hash maps. Hash maps
// * support custom initial table sizes (rounded up internally to power-of-2)
// * support custom static allocator objects
// * auto-resize, based on a load factor (i.e. a 64-entry .75 load factor hash will resize
//                                        when the 49th element is inserted)
// * are based on open hashing
// * have O(1) contains, erase
//
// Maps have STL-like copying semantics, and properly initialize and destruct copies of objects
//
// There are two forms of map: coalesced and uncoalesced. Coalesced maps keep the entries in the
// initial segment of an array, so are fast to iterate over; however deletion is approximately
// twice as expensive.
//
// HashMap<T>:
//		bool			insert(const Key& k, const Value& v)	O(1) amortized (exponential resize policy)
//		Value &			operator[](const Key& k)				O(1) for existing objects, else O(1) amortized
//		const Entry *	find(const Key& k);						O(1)
//		bool			erase(const T& k);						O(1)
//		uint32_t			size();									constant
//		void			reserve(uint32_t size);					O(MAX(currentOccupancy,size))
//		void			clear();								O(currentOccupancy) (with zero constant for objects
// without
// destructors)
//      Iterator		getIterator();
//
// operator[] creates an entry if one does not exist, initializing with the default constructor.
// CoalescedHashMap<T> does not support getIterator, but instead supports
// 		const Key *getEntries();
//
// Use of iterators:
//
// for(HashMap::Iterator iter = test.getIterator(); !iter.done(); ++iter)
//			myFunction(iter->first, iter->second);

namespace px
{

template <class Key, class Value, class HashFn = PxHash<Key>, class Allocator = PxAllocator>
class PxHashMap : public px::PxHashMapBase<Key, Value, HashFn, Allocator>
{
  public:
	typedef px::PxHashMapBase<Key, Value, HashFn, Allocator> HashMapBase;
	typedef typename HashMapBase::Iterator Iterator;

	PxHashMap(uint32_t initialTableSize = 64, float loadFactor = 0.75f) : HashMapBase(initialTableSize, loadFactor)
	{
	}
	PxHashMap(uint32_t initialTableSize, float loadFactor, const Allocator& alloc)
	: HashMapBase(initialTableSize, loadFactor, alloc)
	{
	}
	PxHashMap(const Allocator& alloc) : HashMapBase(64, 0.75f, alloc)
	{
	}
	Iterator getIterator()
	{
		return Iterator(HashMapBase::mBase);
	}
};

template <class Key, class Value, class HashFn = PxHash<Key>, class Allocator = PxAllocator>
class PxCoalescedHashMap : public px::PxHashMapBase<Key, Value, HashFn, Allocator>
{
  public:
	typedef px::PxHashMapBase<Key, Value, HashFn, Allocator> HashMapBase;

	PxCoalescedHashMap(uint32_t initialTableSize = 64, float loadFactor = 0.75f)
	: HashMapBase(initialTableSize, loadFactor)
	{
	}
	const PxPair<const Key, Value>* getEntries() const
	{
		return HashMapBase::mBase.getEntries();
	}
};
} // namespace px

// ===== 원본: physx/include/foundation/PxHashSet.h
// TODO: make this doxy-format

// This header defines two hash sets. Hash sets
// * support custom initial table sizes (rounded up internally to power-of-2)
// * support custom static allocator objects
// * auto-resize, based on a load factor (i.e. a 64-entry .75 load factor hash will resize
//                                        when the 49th element is inserted)
// * are based on open hashing
//
// Sets have STL-like copying semantics, and properly initialize and destruct copies of objects
//
// There are two forms of set: coalesced and uncoalesced. Coalesced sets keep the entries in the
// initial segment of an array, so are fast to iterate over; however deletion is approximately
// twice as expensive.
//
// HashSet<T>:
//		bool		insert(const T& k)						amortized O(1) (exponential resize policy)
// 		bool		contains(const T& k)	const;			O(1)
//		bool		erase(const T& k);						O(1)
//		uint32_t		size()					const;			constant
//		void		reserve(uint32_t size);					O(MAX(size, currentOccupancy))
//		void		clear();								O(currentOccupancy) (with zero constant for objects without
// destructors)
//      Iterator    getIterator();
//
// Use of iterators:
//
// for(HashSet::Iterator iter = test.getIterator(); !iter.done(); ++iter)
//			myFunction(*iter);
//
// CoalescedHashSet<T> does not support getIterator, but instead supports
// 		const Key *getEntries();
//
// insertion into a set already containing the element fails returning false, as does
// erasure of an element not in the set
//

namespace px
{
template <class Key, class HashFn = PxHash<Key>, class Allocator = PxAllocator>
class PxHashSet : public px::PxHashSetBase<Key, HashFn, Allocator, false>
{
  public:
	typedef px::PxHashSetBase<Key, HashFn, Allocator, false> HashSetBase;
	typedef typename HashSetBase::Iterator Iterator;

	PxHashSet(uint32_t initialTableSize = 64, float loadFactor = 0.75f) : HashSetBase(initialTableSize, loadFactor)
	{
	}
	PxHashSet(uint32_t initialTableSize, float loadFactor, const Allocator& alloc)
	: HashSetBase(initialTableSize, loadFactor, alloc)
	{
	}
	PxHashSet(const Allocator& alloc) : HashSetBase(64, 0.75f, alloc)
	{
	}
	Iterator getIterator()
	{
		return Iterator(HashSetBase::mBase);
	}
};

template <class Key, class HashFn = PxHash<Key>, class Allocator = PxAllocator>
class PxCoalescedHashSet : public px::PxHashSetBase<Key, HashFn, Allocator, true>
{
  public:
	typedef typename px::PxHashSetBase<Key, HashFn, Allocator, true> HashSetBase;

	PxCoalescedHashSet(uint32_t initialTableSize = 64, float loadFactor = 0.75f)
	: HashSetBase(initialTableSize, loadFactor)
	{
	}

	PxCoalescedHashSet(uint32_t initialTableSize, float loadFactor, const Allocator& alloc)
	: HashSetBase(initialTableSize, loadFactor, alloc)
	{
	}
	PxCoalescedHashSet(const Allocator& alloc) : HashSetBase(64, 0.75f, alloc)
	{
	}

	const Key* getEntries() const
	{
		return HashSetBase::mBase.getEntries();
	}
};

} // namespace px

// ===== 원본: physx/include/foundation/PxBitMap.h
namespace px
{
	/*!
	Allocation helper with cookie parameter that can be overloaded for pinned memory support
	*/
	template <class AllocT>
	EHDI void* PxBitMapAlloc(AllocT& a, PxU32 bytes, const char* file, int line, uint32_t* /*cookie*/)
	{
		return a.allocate(bytes, file, line);
	}

	/*!
	Deallocation helper with cookie parameter that can be overloaded for pinned memory support
	*/
	template <class AllocT>
	EHDI void PxBitMapDealloc(AllocT& a, void* ptr, uint32_t* /*cookie*/)
	{
		a.deallocate(ptr);
	}

	/*!
	Hold a bitmap with operations to set,reset or test given bit.

	We inhibit copy to prevent unintentional copies. If a copy is desired copy() should be used or
	alternatively a copy constructor implemented.
	*/
	template<class PxAllocator>
	class PxBitMapBase : public PxUserAllocated
	{
		E_NOCOPY(PxBitMapBase)

	public:

		// EPX_SERIALIZATION
		/* todo: explicit */ PxBitMapBase(const PxEMPTY)
		{
			if(mMap)
				mWordCount |= EPX_SIGN_BITMASK;
		}
		//~EPX_SERIALIZATION

		EHDI PxBitMapBase(const PxAllocator& allocator) : mMap(0), mWordCount(0), mAllocator(allocator) {}

		EHDI PxBitMapBase() : mMap(0), mWordCount(0) {}

		EHDI ~PxBitMapBase()
		{
			release();
		}

		EHDI void release()
		{
			if(mMap && !isInUserMemory())
				PxBitMapDealloc(mAllocator, mMap, NULL);
			mMap = NULL;
		}

		EHD PxAllocator&	getAllocator() { return mAllocator; }

		EHDI void growAndSet(PxU32 index)
		{
			extend(index + 1);
			mMap[index >> 5] |= 1 << (index & 31);
		}

		EHDI void growAndReset(PxU32 index)
		{
			extend(index + 1);
			mMap[index >> 5] &= ~(1 << (index & 31));
		}

		EHDI PxIntBool boundedTest(PxU32 index) const
		{
			return PxIntBool(index >> 5 >= getWordCount() ? PxIntFalse : (mMap[index >> 5] & (1 << (index & 31))));
		}

		EHDI void boundedReset(PxU32 index)
		{
			if((index >> 5) < getWordCount())
				mMap[index >> 5] &= ~(1 << (index & 31));
		}

		// Special optimized versions, when you _know_ your index is in range
		EHDI void set(PxU32 index)
		{
			E_ASSERT(index<getWordCount() * 32);
			mMap[index >> 5] |= 1 << (index & 31);
		}

		EHDI void reset(PxU32 index)
		{
			E_ASSERT(index<getWordCount() * 32);
			mMap[index >> 5] &= ~(1 << (index & 31));
		}

		EHDI PxIntBool test(PxU32 index) const
		{
			E_ASSERT(index<getWordCount() * 32);
			return PxIntBool(mMap[index >> 5] & (1 << (index & 31)));
		}

		// nibble == 4 bits
		EHDI PxU32 getNibbleFast(PxU32 nibIndex) const
		{
			const PxU32 bitIndex = nibIndex << 2;
			E_ASSERT(bitIndex < getWordCount() * 32);
			return (mMap[bitIndex >> 5] >> (bitIndex & 31)) & 0xf;
		}

		EHDI void andNibbleFast(PxU32 nibIndex, PxU32 mask)
		{
			//TODO: there has to be a faster way...
			const PxU32 bitIndex = nibIndex << 2;
			const PxU32 shift = (bitIndex & 31);
			const PxU32 nibMask = (0xfu << shift);

			E_ASSERT(bitIndex < getWordCount() * 32);

			mMap[bitIndex >> 5] &= ((mask << shift) | ~nibMask);
		}

		EHDI void orNibbleFast(PxU32 nibIndex, PxU32 mask)
		{
			E_ASSERT(!(mask & ~0xfu)); //check extra bits are not set

			const PxU32 bitIndex = nibIndex << 2;
			const PxU32 shift = bitIndex & 31;

			E_ASSERT(bitIndex < getWordCount() * 32);

			mMap[bitIndex >> 5] |= (mask << shift);
		}

		void clear()
		{
			PxMemSet(mMap, 0, getWordCount() * sizeof(PxU32));
		}

		void resizeAndClear(PxU32 newBitCount)
		{
			extendUninitialized(newBitCount);
			PxMemSet(mMap, 0, getWordCount() * sizeof(PxU32));
		}

		void setEmpty()
		{
			mMap = NULL;
			mWordCount = 0;
		}

		void setWords(PxU32* map, PxU32 wordCount)
		{
			mMap = map;
			mWordCount = wordCount | EPX_SIGN_BITMASK;
		}

		// !!! only sets /last/ bit to value
		void resize(PxU32 newBitCount, bool value = false)
		{
			E_ASSERT(!value); // only new class supports this
			E_UNUSED(value);
			extend(newBitCount);
		}

		EHD	PxU32 size() const { return getWordCount() * 32; }

		void copy(const PxBitMapBase& a)
		{
			extendUninitialized(a.getWordCount() << 5);
			PxMemCopy(mMap, a.mMap, a.getWordCount() * sizeof(PxU32));
			if(getWordCount() > a.getWordCount())
				PxMemSet(mMap + a.getWordCount(), 0, (getWordCount() - a.getWordCount()) * sizeof(PxU32));
		}

		EHDI PxU32 count()		const
		{
			// NOTE: we can probably do this faster, since the last steps in PxBitCount can be defered to
			// the end of the seq. + 64/128bits at a time + native bit counting instructions(360 is fast non micro code).
			PxU32 count = 0;
			const PxU32 wordCount = getWordCount();
			for(PxU32 i = 0; i<wordCount; i++)
				count += PxBitCount(mMap[i]);

			return count;
		}

		EHDI PxU32 count(PxU32 start, PxU32 length) const
		{
			const PxU32 end = PxMin(getWordCount() << 5, start + length);
			PxU32 count = 0;
			for(PxU32 i = start; i<end; i++)
				count += (test(i) != 0);
			return count;
		}

		//! returns 0 if no bits set (!!!)
		PxU32 findLast() const
		{
			const PxU32 wordCount = getWordCount();
			for(PxU32 i = wordCount; i-- > 0;)
			{
				if(mMap[i])
					return (i << 5) + PxHighestSetBit(mMap[i]);
			}
			return PxU32(0);
		}

		bool hasAnyBitSet() const
		{
			const PxU32 wordCount = getWordCount();
			for(PxU32 i = 0; i<wordCount; i++)
			{
				if (mMap[i])
					return true;
			}
			return false;
		}

		// the obvious combiners and some used in the SDK

		struct OR { EHDI PxU32 operator()(PxU32 a, PxU32 b) { return a | b; } };
		struct AND { EHDI PxU32 operator()(PxU32 a, PxU32 b) { return a&b; } };
		struct XOR { EHDI PxU32 operator()(PxU32 a, PxU32 b) { return a^b; } };

		// we use auxiliary functions here so as not to generate combiners for every combination
		// of allocators

		template<class Combiner, class _>
		EHDI void combineInPlace(const PxBitMapBase<_>& b)
		{
			combine1<Combiner>(b.mMap, b.getWordCount());
		}

		template<class Combiner, class _1, class _2>
		EHDI void combine(const PxBitMapBase<_1>& a, const PxBitMapBase<_2>& b)
		{
			combine2<Combiner>(a.mMap, a.getWordCount(), b.mMap, b.getWordCount());
		}

		EHD const PxU32*	getWords()			const	{ return mMap; }
		EHD PxU32*			getWords()					{ return mMap; }

		// EPX_SERIALIZATION
		EHD PxU32			getWordCount()		const	{ return mWordCount & ~EPX_SIGN_BITMASK; }

		// We need one bit to mark arrays that have been deserialized from a user-provided memory block.
		EHD	PxU32			isInUserMemory()	const	{ return mWordCount & EPX_SIGN_BITMASK; }
		//~EPX_SERIALIZATION

		/*!
		Iterate over indices in a bitmap

		This iterator is good because it finds the set bit without looping over the cached bits upto 31 times.
		However it does require a variable shift.
		*/
		class Iterator
		{
		public:
			static const PxU32 DONE = 0xffffffff;

			EHDI Iterator(const PxBitMapBase &map) : mBitMap(map)
			{
				reset();
			}

			EHDI Iterator& operator=(const Iterator& other)
			{
				E_ASSERT(&mBitMap == &other.mBitMap);
				mBlock = other.mBlock;
				mIndex = other.mIndex;
				return *this;
			}

			EHDI PxU32	getNext()
			{
				if(mBlock)
				{
					PxU32 block = mBlock;
					PxU32 index = mIndex;

					const PxU32 bitIndex = index << 5 | PxLowestSetBit(block);
					block &= block - 1;
					PxU32 wordCount = mBitMap.getWordCount();
					while(!block && ++index < wordCount)
						block = mBitMap.mMap[index];

					mBlock = block;
					mIndex = index;

					return bitIndex;
				}
				return DONE;
			}

			EHDI void reset()
			{
				PxU32 index = 0;
				PxU32 block = 0;

				PxU32 wordCount = mBitMap.getWordCount();
				while(index < wordCount && ((block = mBitMap.mMap[index]) == 0))
					++index;

				mBlock = block;
				mIndex = index;
			}
		private:
			PxU32 mBlock, mIndex;
			const PxBitMapBase& mBitMap;
		};

		// DS: faster but less general: hasBits() must be true or getNext() is illegal so it is the calling code's responsibility to ensure that getNext() is not called illegally.
		class PxLoopIterator
		{
			E_NOCOPY(PxLoopIterator)

		public:
			EHD PxLoopIterator(const PxBitMapBase &map) : mMap(map.getWords()), mBlock(0), mIndex(-1), mWordCount(PxI32(map.getWordCount())) {}

			EHD bool hasBits()
			{
				E_ASSERT(mIndex<mWordCount);
				while (mBlock == 0)
				{
					if (++mIndex == mWordCount)
						return false;
					mBlock = mMap[mIndex];
				}
				return true;
			}

			EHD PxU32 getNext()
			{
				E_ASSERT(mIndex<mWordCount && mBlock != 0);
				PxU32 result = PxU32(mIndex) << 5 | PxLowestSetBit(mBlock);	// will assert if mask is zero
				mBlock &= (mBlock - 1);
				return result;
			}

		private:
			const PxU32*const mMap;
			PxU32 mBlock;		// the word we're currently scanning
			PxI32 mIndex;		// the index of the word we're currently looking at
			PxI32 mWordCount;
		};

		//Class to iterate over the bitmap from a particular start location rather than the beginning of the list
		class PxCircularIterator
		{
		public:
			static const PxU32 DONE = 0xffffffff;

			EHDI PxCircularIterator(const PxBitMapBase &map, PxU32 index) : mBitMap(map)
			{
				PxU32 localIndex = 0;
				PxU32 startIndex = 0;

				const PxU32 wordCount = mBitMap.getWordCount();
				if((index << 5) < wordCount)
				{
					localIndex = index << 5;
					startIndex = localIndex;
				}

				PxU32 block = 0;
				if(localIndex < wordCount)
				{
					block = mBitMap.mMap[localIndex];
					if(block == 0)
					{
						localIndex = (localIndex + 1) % wordCount;
						while(localIndex != startIndex && (block = mBitMap.mMap[localIndex]) == 0)
							localIndex = (localIndex + 1) % wordCount;
					}
				}

				mIndex = localIndex;
				mBlock = block;
				mStartIndex = startIndex;
			}

			EHDI PxU32	getNext()
			{
				if(mBlock)
				{
					PxU32 index = mIndex;
					PxU32 block = mBlock;
					const PxU32 startIndex = mStartIndex;

					PxU32 bitIndex = index << 5 | PxLowestSetBit(block);
					block &= block - 1;
					PxU32 wordCount = mBitMap.getWordCount();
					while (!block && (index = ((index + 1) % wordCount)) != startIndex)
						block = mBitMap.mMap[index];

					mIndex = index;
					mBlock = block;

					return bitIndex;
				}
				return DONE;
			}

		private:
			PxU32 mBlock, mIndex;
			PxU32 mStartIndex;
			const PxBitMapBase& mBitMap;

			E_NOCOPY(PxCircularIterator)
		};

	protected:
		PxU32*		mMap;			//one bit per index
		PxU32		mWordCount;
		PxAllocator	mAllocator;
		PxU8		mPadding[3];	// PT: "mAllocator" is empty but consumes 1 byte

		void extend(PxU32 size)
		{
			const PxU32 newWordCount = (size + 31) >> 5;
			if (newWordCount > getWordCount())
			{
				uint32_t cookie = 0;
				void* newMap = PxBitMapAlloc(mAllocator, newWordCount * sizeof(PxU32), EPX_FL, &cookie);
				if (mMap)
				{
					PxMemCopy(newMap, mMap, getWordCount() * sizeof(PxU32));
					if (!isInUserMemory())
						PxBitMapDealloc(mAllocator, mMap, &cookie);
				}
				mMap = reinterpret_cast<PxU32*>(newMap);
				PxMemSet(mMap + getWordCount(), 0, (newWordCount - getWordCount()) * sizeof(PxU32));
				// also resets the isInUserMemory bit
				mWordCount = newWordCount;
			}
		}

		void extendUninitialized(PxU32 size)
		{
			PxU32 newWordCount = (size + 31) >> 5;
			if (newWordCount > getWordCount())
			{
				if (mMap && !isInUserMemory())
					PxBitMapDealloc(mAllocator, mMap, NULL);
				// also resets the isInUserMemory bit
				mWordCount = newWordCount;
				void* newMap = PxBitMapAlloc(mAllocator, mWordCount * sizeof(PxU32), EPX_FL, NULL);
				mMap = reinterpret_cast<PxU32*>(newMap);
			}
		}

		template<class Combiner>
		void combine1(const PxU32* words, PxU32 length)
		{
			extend(length << 5);
			PxU32 combineLength = PxMin(getWordCount(), length);
			for (PxU32 i = 0; i<combineLength; i++)
				mMap[i] = Combiner()(mMap[i], words[i]);
		}

		template<class Combiner>
		void combine2(const PxU32* words1, PxU32 length1,
			const PxU32* words2, PxU32 length2)
		{
			extendUninitialized(PxMax(length1, length2) << 5);

			PxU32 commonSize = PxMin(length1, length2);

			for (PxU32 i = 0; i<commonSize; i++)
				mMap[i] = Combiner()(words1[i], words2[i]);

			for (PxU32 i = commonSize; i<length1; i++)
				mMap[i] = Combiner()(words1[i], 0);

			for (PxU32 i = commonSize; i<length2; i++)
				mMap[i] = Combiner()(0, words2[i]);
		}

		friend class Iterator;
	};

	typedef PxBitMapBase<PxAllocator> PxBitMap;

} // namespace px

// ===== 원본: physx/include/foundation/PxSortInternals.h
namespace px
{
template <class T, class Predicate>
EHDI void PxMedian3(T* elements, int32_t first, int32_t last, Predicate& compare)
{
	/*
	This creates sentinels because we know there is an element at the start minimum(or equal)
	than the pivot and an element at the end greater(or equal) than the pivot. Plus the
	median of 3 reduces the chance of degenerate behavour.
	*/

	int32_t mid = (first + last) / 2;

	if(compare(elements[mid], elements[first]))
		PxSwap(elements[first], elements[mid]);

	if(compare(elements[last], elements[first]))
		PxSwap(elements[first], elements[last]);

	if(compare(elements[last], elements[mid]))
		PxSwap(elements[mid], elements[last]);

	// keep the pivot at last-1
	PxSwap(elements[mid], elements[last - 1]);
}

template <class T, class Predicate>
EHDI int32_t PxPartition(T* elements, int32_t first, int32_t last, Predicate& compare)
{
	PxMedian3(elements, first, last, compare);

	/*
	WARNING: using the line:

	T partValue = elements[last-1];

	and changing the scan loops to:

	while(comparator.greater(partValue, elements[++i]));
	while(comparator.greater(elements[--j], partValue);

	triggers a compiler optimizer bug on xenon where it stores a double to the stack for partValue
	then loads it as a single...:-(
	*/

	int32_t i = first;    // we know first is less than pivot(but i gets pre incremented)
	int32_t j = last - 1; // pivot is in last-1 (but j gets pre decremented)

	for(;;)
	{
		while(compare(elements[++i], elements[last - 1]))
			;
		while(compare(elements[last - 1], elements[--j]))
			;

		if(i >= j)
			break;

		E_ASSERT(i <= last && j >= first);
		PxSwap(elements[i], elements[j]);
	}
	// put the pivot in place

	E_ASSERT(i <= last && first <= (last - 1));
	PxSwap(elements[i], elements[last - 1]);

	return i;
}

template <class T, class Predicate>
EHDI void PxSmallSort(T* elements, int32_t first, int32_t last, Predicate& compare)
{
	// selection sort - could reduce to fsel on 360 with floats.

	for(int32_t i = first; i < last; i++)
	{
		int32_t m = i;
		for(int32_t j = i + 1; j <= last; j++)
			if(compare(elements[j], elements[m]))
				m = j;

		if(m != i)
			PxSwap(elements[m], elements[i]);
	}
}

template <class PxAllocator>
class PxStack
{
	PxAllocator mAllocator;
	uint32_t mSize, mCapacity;
	int32_t* mMemory;
	bool mRealloc;

  public:
	PxStack(int32_t* memory, uint32_t capacity, const PxAllocator& inAllocator)
	: mAllocator(inAllocator), mSize(0), mCapacity(capacity), mMemory(memory), mRealloc(false)
	{
	}
	~PxStack()
	{
		if(mRealloc)
			mAllocator.deallocate(mMemory);
	}

	void grow()
	{
		mCapacity *= 2;
		int32_t* newMem =
		    reinterpret_cast<int32_t*>(mAllocator.allocate(sizeof(int32_t) * mCapacity, EPX_FL));
		intrinsics::memCopy(newMem, mMemory, mSize * sizeof(int32_t));
		if(mRealloc)
			mAllocator.deallocate(mMemory);
		mRealloc = true;
		mMemory = newMem;
	}

	EHDI void push(int32_t start, int32_t end)
	{
		if(mSize >= mCapacity - 1)
			grow();
		mMemory[mSize++] = start;
		mMemory[mSize++] = end;
	}

	EHDI void pop(int32_t& start, int32_t& end)
	{
		E_ASSERT(!empty());
		end = mMemory[--mSize];
		start = mMemory[--mSize];
	}

	EHDI bool empty()
	{
		return mSize == 0;
	}
};

} // namespace px

// ===== 원본: physx/include/foundation/PxSort.h
#define EPX_SORT_PARANOIA EPX_DEBUG

/**
\brief Sorts an array of objects in ascending order, assuming
that the predicate implements the < operator:

\see PxLess, PxGreater
*/


namespace px
{
template <class T, class Predicate, class PxAllocator>
void PxSort(T* elements, uint32_t count, const Predicate& compare, const PxAllocator& inAllocator,
          const uint32_t initialStackSize = 32)
{
	static const uint32_t SMALL_SORT_CUTOFF = 5; // must be >= 3 since we need 3 for median

	EPX_ALLOCA(stackMem, int32_t, initialStackSize);
	PxStack<PxAllocator> stack(stackMem, initialStackSize, inAllocator);

	int32_t first = 0, last = int32_t(count - 1);
	if(last > first)
	{
		for(;;)
		{
			while(last > first)
			{
				E_ASSERT(first >= 0 && last < int32_t(count));
				if(uint32_t(last - first) < SMALL_SORT_CUTOFF)
				{
					PxSmallSort(elements, first, last, compare);
					break;
				}
				else
				{
					const int32_t partIndex = PxPartition(elements, first, last, compare);

					// push smaller sublist to minimize stack usage
					if((partIndex - first) < (last - partIndex))
					{
						stack.push(first, partIndex - 1);
						first = partIndex + 1;
					}
					else
					{
						stack.push(partIndex + 1, last);
						last = partIndex - 1;
					}
				}
			}

			if(stack.empty())
				break;

			stack.pop(first, last);
		}
	}
	for(uint32_t i = 1; i < count; i++)
		E_ASSERT(!compare(elements[i], elements[i - 1]));
}

template <class T, class Predicate>
void PxSort(T* elements, uint32_t count, const Predicate& compare)
{
	PxSort(elements, count, compare, typename PxAllocatorTraits<T>::Type());
}

template <class T>
void PxSort(T* elements, uint32_t count)
{
	PxSort(elements, count, PxLess<T>(), typename PxAllocatorTraits<T>::Type());
}

} // namespace px

// ===== 원본: physx/source/lowlevelaabb/include/BpVolumeData.h
namespace px
{
	namespace Bp
	{
		typedef PxU32 AggregateHandle;	// PT: currently an index in mAggregates array

		struct ElementType
		{
			enum Enum
			{
				eSHAPE = 0,
				eTRIGGER,

				eCOUNT
			};
		};
		E_STATIC_ASSERT(ElementType::eCOUNT <= 4);			// 2 bits reserved for type

		#define EPX_CUDA_INLINE	EHD

		struct VolumeData
		{
			EPX_CUDA_INLINE	void				reset()
												{
													mAggregate = EPX_INVALID_U32;
													mUserData = NULL;
												}

			EPX_CUDA_INLINE	void				setSingleActor()		{ mAggregate = EPX_INVALID_U32;			}
			EPX_CUDA_INLINE	bool				isSingleActor()	const	{ return mAggregate == EPX_INVALID_U32;	}

			EPX_CUDA_INLINE	void				setUserData(void* userData)
												{
//													E_ASSERT(!(size_t(userData) & 3));
													mUserData = userData;
												}

			EPX_CUDA_INLINE void*				getUserData() const
												{
													return reinterpret_cast<void*>(size_t(mUserData)& (~size_t(3)));
												}

			EPX_CUDA_INLINE void					setVolumeType(ElementType::Enum volumeType)
												{
													E_ASSERT(volumeType < 2);
													mUserData = reinterpret_cast<void*>(size_t(getUserData()) | size_t(volumeType));
												}

			EPX_CUDA_INLINE ElementType::Enum	getVolumeType() const
												{
													return ElementType::Enum(size_t(mUserData) & 3);
												}

			EPX_CUDA_INLINE	void				setAggregate(AggregateHandle handle)
												{
													E_ASSERT(handle != EPX_INVALID_U32);
													mAggregate = (handle << 1) | 1;
												}

			EPX_CUDA_INLINE	bool				isAggregate()	const { return !isSingleActor() && ((mAggregate & 1) != 0); }

			EPX_CUDA_INLINE	void				setAggregated(AggregateHandle handle)
												{
													E_ASSERT(handle != EPX_INVALID_U32);
													mAggregate = (handle << 1) | 0;
												}

			EPX_CUDA_INLINE	bool				isAggregated()	const
												{
													return !isSingleActor() && ((mAggregate & 1) == 0);
												}

			EPX_CUDA_INLINE	AggregateHandle		getAggregateOwner()	const { return mAggregate >> 1; }
			EPX_CUDA_INLINE	AggregateHandle		getAggregate()		const { return mAggregate >> 1; }

		private:
							void*				mUserData;	// PT: in PhysX this is an Sc::ElementSim ptr
			// PT: TODO: consider moving this to a separate array, which wouldn't be allocated at all for people not using aggregates.
			// PT: current encoding:
			// aggregate == EPX_INVALID_U32 => single actor
			// aggregate != EPX_INVALID_U32 => aggregate index<<1|LSB. LSB==1 for aggregates, LSB==0 for aggregated actors.
							AggregateHandle		mAggregate;
		};

	}
}

// ===== 원본: physx/source/lowlevelaabb/src/BpBroadPhaseMBPCommon.h
namespace px
{
namespace Bp
{

#define MBP_USE_WORDS
#define MBP_USE_NO_CMP_OVERLAP
	#define MBP_SIMD_OVERLAP

	typedef	PxU16	MBP_Index;
	typedef	PxU32	MBP_ObjectIndex;	// PT: index in mMBP_Objects
	typedef	PxU32	MBP_Handle;			// PT: returned to MBP users, combination of index/flip-flop/static-bit

	struct IAABB : public PxUserAllocated
	{
		EHD bool isInside(const IAABB& box) const
		{
			if(box.mMinX>mMinX)	return false;
			if(box.mMinY>mMinY)	return false;
			if(box.mMinZ>mMinZ)	return false;
			if(box.mMaxX<mMaxX)	return false;
			if(box.mMaxY<mMaxY)	return false;
			if(box.mMaxZ<mMaxZ)	return false;
			return true;
		}

		EHD			PxIntBool		intersects(const IAABB& a)	const
		{
			if(mMaxX < a.mMinX || a.mMaxX < mMinX
			|| mMaxY < a.mMinY || a.mMaxY < mMinY
			|| mMaxZ < a.mMinZ || a.mMaxZ < mMinZ
			)
				return PxIntFalse;
			return PxIntTrue;
		}

		EHD			PxIntBool		intersectNoTouch(const IAABB& a)	const
		{
			if(mMaxX <= a.mMinX || a.mMaxX <= mMinX
			|| mMaxY <= a.mMinY || a.mMaxY <= mMinY
			|| mMaxZ <= a.mMinZ || a.mMaxZ <= mMinZ
			)
				return PxIntFalse;
			return PxIntTrue;
		}

		EHD	void	initFrom2(const PxBounds3& box)
		{
			const PxU32*  binary = reinterpret_cast<const PxU32*>(&box.minimum.x);
			mMinX = encodeFloat(binary[0])>>1;
			mMinY = encodeFloat(binary[1])>>1;
			mMinZ = encodeFloat(binary[2])>>1;
			mMaxX = encodeFloat(binary[3])>>1;
			mMaxY = encodeFloat(binary[4])>>1;
			mMaxZ = encodeFloat(binary[5])>>1;
		}

		EHD	void	decode(PxBounds3& box)	const
		{
			PxU32*  binary = reinterpret_cast<PxU32*>(&box.minimum.x);
			binary[0] = decodeFloat(mMinX<<1);
			binary[1] = decodeFloat(mMinY<<1);
			binary[2] = decodeFloat(mMinZ<<1);
			binary[3] = decodeFloat(mMaxX<<1);
			binary[4] = decodeFloat(mMaxY<<1);
			binary[5] = decodeFloat(mMaxZ<<1);
		}

		EHD PxU32	getMin(PxU32 i)	const	{	return (&mMinX)[i];	}
		EHD PxU32	getMax(PxU32 i)	const	{	return (&mMaxX)[i];	}

		PxU32 mMinX;
		PxU32 mMinY;
		PxU32 mMinZ;
		PxU32 mMaxX;
		PxU32 mMaxY;
		PxU32 mMaxZ;
	};

	struct SIMD_AABB : public PxUserAllocated
	{
		EHD	void	initFrom(const PxBounds3& box)
		{
			const PxU32*  binary = reinterpret_cast<const PxU32*>(&box.minimum.x);
			mMinX = encodeFloat(binary[0]);
			mMinY = encodeFloat(binary[1]);
			mMinZ = encodeFloat(binary[2]);
			mMaxX = encodeFloat(binary[3]);
			mMaxY = encodeFloat(binary[4]);
			mMaxZ = encodeFloat(binary[5]);
		}

		EHD	void	initFrom2(const PxBounds3& box)
		{
			const PxU32*  binary = reinterpret_cast<const PxU32*>(&box.minimum.x);
			mMinX = encodeFloat(binary[0])>>1;
			mMinY = encodeFloat(binary[1])>>1;
			mMinZ = encodeFloat(binary[2])>>1;
			mMaxX = encodeFloat(binary[3])>>1;
			mMaxY = encodeFloat(binary[4])>>1;
			mMaxZ = encodeFloat(binary[5])>>1;
		}

		EHD	void	decode(PxBounds3& box)	const
		{
			PxU32*  binary = reinterpret_cast<PxU32*>(&box.minimum.x);
			binary[0] = decodeFloat(mMinX<<1);
			binary[1] = decodeFloat(mMinY<<1);
			binary[2] = decodeFloat(mMinZ<<1);
			binary[3] = decodeFloat(mMaxX<<1);
			binary[4] = decodeFloat(mMaxY<<1);
			binary[5] = decodeFloat(mMaxZ<<1);
		}

		EHD bool isInside(const SIMD_AABB& box) const
		{
			if(box.mMinX>mMinX)	return false;
			if(box.mMinY>mMinY)	return false;
			if(box.mMinZ>mMinZ)	return false;
			if(box.mMaxX<mMaxX)	return false;
			if(box.mMaxY<mMaxY)	return false;
			if(box.mMaxZ<mMaxZ)	return false;
			return true;
		}

		EHD			PxIntBool		intersects(const SIMD_AABB& a)	const
		{
			if(mMaxX < a.mMinX || a.mMaxX < mMinX
			|| mMaxY < a.mMinY || a.mMaxY < mMinY
			|| mMaxZ < a.mMinZ || a.mMaxZ < mMinZ
			)
				return PxIntFalse;
			return PxIntTrue;
		}

		EHD			PxIntBool		intersectNoTouch(const SIMD_AABB& a)	const
		{
			if(mMaxX <= a.mMinX || a.mMaxX <= mMinX
			|| mMaxY <= a.mMinY || a.mMaxY <= mMinY
			|| mMaxZ <= a.mMinZ || a.mMaxZ <= mMinZ
			)
				return PxIntFalse;
			return PxIntTrue;
		}

		PxU32 mMinX;
		PxU32 mMaxX;
		PxU32 mMinY;
		PxU32 mMinZ;
		PxU32 mMaxY;
		PxU32 mMaxZ;
	};

}
} // namespace px

// ===== 원본: physx/source/lowlevelaabb/include/BpAABBManagerTasks.h
namespace px
{
namespace Bp
{
	class AABBManager;
	class Aggregate;

	class AggregateBoundsComputationTask : public Cm::Task, public PxUserAllocated
	{
		E_NOCOPY(AggregateBoundsComputationTask)
		public:
								AggregateBoundsComputationTask(PxU64 contextId) :
									Cm::Task	(contextId),
									mManager	(NULL),
									mStart		(0),
									mNbToGo		(0),
									mAggregates	(NULL)
								{}
								~AggregateBoundsComputationTask()	{}

		virtual const char*		getName() const { return "AggregateBoundsComputationTask"; }
		virtual void			runInternal();

				void			Init(AABBManager* manager, PxU32 start, PxU32 nb, Aggregate** aggregates)
								{
									mManager	= manager;
									mStart		= start;
									mNbToGo		= nb;
									mAggregates	= aggregates;
								}
		private:
				AABBManager*	mManager;
				PxU32			mStart;
				PxU32			mNbToGo;
				Aggregate**		mAggregates;
	};

	class PreBpUpdateTask : public Cm::Task, public PxUserAllocated
	{
		E_NOCOPY(PreBpUpdateTask)
	public:
								PreBpUpdateTask(PxU64 contextId) : Cm::Task(contextId), mManager(NULL), mNumCpuTasks(0)	{}
								~PreBpUpdateTask() {}

		virtual const char*		getName() const { return "PreBpUpdateTask"; }
		virtual void			runInternal();

				void			Init(AABBManager* manager, PxU32 numCpuTasks)
								{
									mManager = manager;
									mNumCpuTasks = numCpuTasks;
								}
	private:
				AABBManager*	mManager;
				PxU32			mNumCpuTasks;
	};

}
} //namespace px

// ===== 원본: physx/source/lowlevelaabb/include/BpAABBManagerBase.h
namespace px
{
class PxcScratchAllocator;
class PxRenderOutput;
class PxBaseTask;

namespace Cm
{
	class FlushPool;
}

namespace Bp
{
	typedef PxU32 BoundsIndex;
	//typedef PxU32 ActorHandle;

	/**
	\brief Changes to the configuration of overlap pairs are reported as void* pairs.
	\note Each void* in the pair corresponds to the void* passed to AABBManager::createVolume.
	\see AABBManager::createVolume, AABBManager::getCreatedOverlaps, AABBManager::getDestroyedOverlaps
	*/
	struct AABBOverlap
	{
		EHD AABBOverlap() {}
		EHD AABBOverlap(void* userData0, void* userData1/*, ActorHandle pairHandle*/) : mUserData0(userData0), mUserData1(userData1)/*, mPairHandle(pairHandle)*/ 
		{
			// PT: TODO: why is this forbidden?
			E_ASSERT(userData0 != userData1);
		}

		// PT: these will eventually be the userData pointers passed to addBounds(), i.e. Sc::ElementSim pointers in PhysX. This may not be
		// necessary at all, since in the current design the bounds indices themselves come from BP clients (they're not handles managed by the BP).
		// So there's a 1:1 mapping between bounds-indices (which are effectively edlement IDs in PhysX) and the userData pointers (Sc::ElementSim).
		// Thus we could just return bounds indices directly to users - at least in the context of PhysX, maybe the standalone BP is different.
		void*	mUserData0;
		void*	mUserData1;

		// PT: TODO: not sure what happened there but mPairUserData is not used inside the BP itself so we need to revisit this.
		/*		union
				{
					ActorHandle	mPairHandle;		//For created pairs, this is the index into the pair in the pair manager
					void*		mUserData;			//For deleted pairs, this is the user data written by the application to the pair
				};*/
		void*	mPairUserData;			//For deleted pairs, this is the user data written by the application to the pair
	};

	struct BpCacheData : public PxSListEntry
	{
		PxArray<AABBOverlap> mCreatedPairs[2];
		PxArray<AABBOverlap> mDeletedPairs[2];

		void reset()
		{
			mCreatedPairs[0].resizeUninitialized(0);
			mCreatedPairs[1].resizeUninitialized(0);
			mDeletedPairs[0].resizeUninitialized(0);
			mDeletedPairs[1].resizeUninitialized(0);
		}
	};

	typedef	PxPinnedArraySafe<Bp::FilterGroup::Enum>	GroupsArrayPinnedSafe;
	typedef	PxPinnedArraySafe<VolumeData>				VolumeDataArrayPinnedSafe;
	typedef	PxPinnedArraySafe<ShapeHandle>				ShapeHandleArrayPinnedSafe;

	class BoundsArray : public PxUserAllocated
	{
												E_NOCOPY(BoundsArray)
	public:
												BoundsArray(PxVirtualAllocator& allocator) : mBounds(allocator), mHasAnythingChanged(true)	{} //needs to be set explicitly for PxgBounds first copy

		virtual									~BoundsArray(){}

		EHD	void					initEntry(PxU32 index)
												{
													index++;	// PT: always pretend we need one more entry, to make sure reading the last used entry will be SIMD-safe.
													const PxU32 oldCapacity = mBounds.capacity();
													if (index >= oldCapacity)
													{
														const PxU32 newCapacity = PxNextPowerOfTwo(index);
														mBounds.reserve(newCapacity);
														mBounds.forceSize_Unsafe(newCapacity);
													}
												}

		virtual void							updateBounds(const PxTransform& transform, const PxGeometry& geom, PxU32 index, PxU32 /*indexFrom*/)
												{
													Gu::computeBounds(mBounds[index], geom, transform, 0.0f, 1.0f);
													mHasAnythingChanged = true;
												}


		virtual void							setBounds(const PxBounds3& bounds, PxU32 index)
												{
	//												EPX_CHECK_AND_RETURN(bounds.isValid() && !bounds.isEmpty(), "BoundsArray::setBounds - illegal bounds\n");
													mBounds[index] = bounds;
													mHasAnythingChanged = true;
												}

		EHD const PxBounds3*		begin()					const	{ return mBounds.begin();		}
		EHD PxBounds3*				begin()							{ return mBounds.begin();		}
		EHD PxBoundsArrayPinned&	getBounds()						{ return mBounds;				}
		EHD	const PxBounds3&		getBounds(PxU32 index)	const	{ return mBounds[index];		}
		EHD PxU32					size()					const	{ return mBounds.size();		}
		EHD	bool					hasChanged()			const	{ return mHasAnythingChanged;	}
		EHD	void					resetChangedState()				{ mHasAnythingChanged = false;	}
		EHD	void					setChangedState()				{ mHasAnythingChanged = true;	}

						void					shiftOrigin(const PxVec3& shift)
												{
													// we shift some potential NaNs here because we don't know what's active, but should be harmless
													const PxU32 nbBounds = mBounds.size();
													for(PxU32 i=0; i<nbBounds; i++)
													{
														mBounds[i].minimum -= shift;
														mBounds[i].maximum -= shift;
													}
													mHasAnythingChanged = true;
												}
	protected:
						PxBoundsArrayPinned		mBounds;
						bool					mHasAnythingChanged;
	};

	/**
	\brief A structure responsible for:
	* storing an aabb representation for each active shape in the related scene
	* managing the creation/removal of aabb representations when their related shapes are created/removed
	* updating all aabbs that require an update due to modification of shape geometry or transform
	* updating the aabb of all aggregates from the union of the aabbs of all shapes that make up each aggregate
	* computing and reporting the incremental changes to the set of overlapping aabb pairs
	*/
	class AABBManagerBase : public PxUserAllocated
	{
												E_NOCOPY(AABBManagerBase)
	public:
												AABBManagerBase(BroadPhase& bp, BoundsArray& boundsArray, PxFloatArrayPinnedSafe& contactDistance,
																PxU32 maxNbAggregates, PxU32 maxNbShapes, PxVirtualAllocator& allocator, PxU64 contextID,
																PxPairFilteringMode::Enum kineKineFilteringMode, PxPairFilteringMode::Enum staticKineFilteringMode);

		virtual									~AABBManagerBase() {}

		virtual			void					destroy() = 0;

		virtual			AggregateHandle			createAggregate(BoundsIndex index, Bp::FilterGroup::Enum group, void* userData, PxU32 maxNumShapes, PxAggregateFilterHint filterHint, PxU32 envID) = 0;
		virtual			bool					destroyAggregate(BoundsIndex& index, Bp::FilterGroup::Enum& group, AggregateHandle aggregateHandle) = 0;

		virtual			bool					addBounds(BoundsIndex index, PxReal contactDistance, Bp::FilterGroup::Enum group, void* userdata, AggregateHandle aggregateHandle, ElementType::Enum volumeType, PxU32 envID) = 0;
		virtual			bool					removeBounds(BoundsIndex index) = 0;

						void					reserveSpaceForBounds(BoundsIndex index);

		EHD	PxIntBool				isMarkedForRemove(BoundsIndex index)	const	{ return mRemovedHandleMap.boundedTest(index);	}
//		EHD	PxIntBool				isMarkedForAdd(BoundsIndex index)		const	{ return mAddedHandleMap.boundedTest(index);	}
		EHD	BroadPhase*				getBroadPhase()							const	{ return &mBroadPhase;							}
		EHD	BoundsArray&			getBoundsArray()								{ return mBoundsArray;							}
		EHD	PxU32					getNbActiveAggregates()					const	{ return mNbAggregates;							}
		EHD	const float*			getContactDistances()					const	{ return mContactDistance.begin();				}
		EHD	PxBitMapPinned&			getChangedAABBMgActorHandleMap()				{ return mChangedHandleMap;						}
		EHD void*					getUserData(const BoundsIndex index)	const	{ return (index<mVolumeData.size()) ? mVolumeData[index].getUserData() : NULL;	}

						void					setContactDistance(BoundsIndex handle, PxReal offset)
												{
													// PT: this works even for aggregated shapes, since the corresponding bit will also be set in the 'updated' map.
													mContactDistance.begin()[handle] = offset;
													setPersistentStateChanged();
													mChangedHandleMap.growAndSet(handle);
												}

						void					setBPGroup(BoundsIndex index, Bp::FilterGroup::Enum group)
												{
													E_ASSERT((index + 1) < mVolumeData.size());
													E_ASSERT(group != Bp::FilterGroup::eINVALID);	// PT: we use group == Bp::FilterGroup::eINVALID to mark removed/invalid entries
													mGroups[index] = group;
												}

		virtual			void					updateBPFirstPass(PxU32 numCpuTasks, Cm::FlushPool& flushPool, bool hasContactDistanceUpdated, PxBaseTask* continuation) = 0;
		virtual			void					updateBPSecondPass(PxcScratchAllocator* scratchAllocator, PxBaseTask* continuation)	= 0;

		virtual			void					postBroadPhase(PxBaseTask*, Cm::FlushPool& flushPool) = 0;
		virtual			void					reallocateChangedAABBMgActorHandleMap(const PxU32 size) = 0;

						AABBOverlap*			getCreatedOverlaps(ElementType::Enum type, PxU32& count)
												{
													E_ASSERT(type < ElementType::eCOUNT);
													count = mCreatedOverlaps[type].size();
													return mCreatedOverlaps[type].begin();
												}

						AABBOverlap*			getDestroyedOverlaps(ElementType::Enum type, PxU32& count)
												{
													E_ASSERT(type < ElementType::eCOUNT);
													count = mDestroyedOverlaps[type].size();
													return mDestroyedOverlaps[type].begin();
												}

						void					freeBuffers();

						struct OutOfBoundsData
						{
							PxU32	mNbOutOfBoundsObjects;
							PxU32	mNbOutOfBoundsAggregates;
							void**	mOutOfBoundsObjects;
							void**	mOutOfBoundsAggregates;
						};
		virtual			bool					getOutOfBoundsObjects(OutOfBoundsData&)		{ return false;	}
		virtual			void					clearOutOfBoundsObjects()					{}

						void					shiftOrigin(const PxVec3& shift);

		virtual			void					visualize(PxRenderOutput& out) = 0;

		virtual			void					releaseDeferredAggregateIds() = 0;
		virtual			void					setGPUStateChanged()			{}
		virtual			void					setPersistentStateChanged()		{}

	


	protected:
						void					reserveShapeSpace(PxU32 nbShapes);

		// PT: we have bitmaps here probably to quickly handle added/removed objects during same frame.
		// PT: TODO: consider replacing with plain arrays (easier to parse, already existing below, etc)
						PxBitMapPinned			mAddedHandleMap;		// PT: indexed by BoundsIndex
						PxBitMapPinned			mRemovedHandleMap;		// PT: indexed by BoundsIndex
						PxBitMapPinned			mChangedHandleMap;

		//Returns true if the bounds was pending insert, false otherwise
		EHD bool					removeBPEntry(BoundsIndex index)	// PT: only for objects passed to the BP
												{
													if (mAddedHandleMap.test(index))		// PT: if object had been added this frame...
													{
														mAddedHandleMap.reset(index);	// PT: ...then simply revert the previous operation locally (it hasn't been passed to the BP yet).
														return true;
													}
													else
														mRemovedHandleMap.set(index);	// PT: else we need to remove it from the BP
													return false;
												}

		EHD void					addBPEntry(BoundsIndex index)
												{
													if (mRemovedHandleMap.test(index))
														mRemovedHandleMap.reset(index);
													else
														mAddedHandleMap.set(index);
												}

		//ML: we create mGroups and mContactDistance in the AABBManager constructor. PxArray will take PxVirtualAllocator as a parameter. Therefore, if GPU BP is using,
		//we will passed a pinned host memory allocator, otherwise, we will just pass a normal allocator.
						GroupsArrayPinnedSafe	mGroups;				// NOTE: we stick Bp::FilterGroup::eINVALID in this slot to indicate that the entry is invalid (removed or never inserted.)
						PxInt32ArrayPinnedSafe	mEnvIDs;				// PT: should ideally be in the GPU class
						PxFloatArrayPinnedSafe&	mContactDistance;
						VolumeDataArrayPinnedSafe	mVolumeData;
						BpFilter				mFilters;

		EHD	void					initEntry(BoundsIndex index, PxReal contactDistance, Bp::FilterGroup::Enum group, void* userData)
												{
													if ((index + 1) >= mVolumeData.size())
														reserveShapeSpace(index + 1);

													// PT: TODO: why is this needed at all? Why aren't size() and capacity() enough?
													mUsedSize = PxMax(index + 1, mUsedSize);

													E_ASSERT(group != Bp::FilterGroup::eINVALID);	// PT: we use group == Bp::FilterGroup::eINVALID to mark removed/invalid entries
													mGroups[index] = group;
													mContactDistance.begin()[index] = contactDistance;
													mVolumeData[index].setUserData(userData);
												}

		EHD	void					initEntry(BoundsIndex index, PxReal contactDistance, Bp::FilterGroup::Enum group, void* userData, ElementType::Enum volumeType)
												{
													initEntry(index, contactDistance, group, userData);
													mVolumeData[index].setVolumeType(volumeType);	// PT: must be done after setUserData
												}

		EHD	void					resetEntry(BoundsIndex index)
												{
													mGroups[index] = Bp::FilterGroup::eINVALID;
													mContactDistance.begin()[index] = 0.0f;
													mVolumeData[index].reset();

													if(index<mEnvIDs.size())
														mEnvIDs[index] = EPX_INVALID_U32;
												}

		// PT: TODO: remove confusion between BoundsIndex and ShapeHandle here!
						ShapeHandleArrayPinnedSafe	mAddedHandles;
						ShapeHandleArrayPinnedSafe	mUpdatedHandles;	// PT: TODO: only on CPU
						ShapeHandleArrayPinnedSafe	mRemovedHandles;

						BroadPhase&				mBroadPhase;
						BoundsArray&			mBoundsArray;

						PxArray<AABBOverlap>	mCreatedOverlaps[ElementType::eCOUNT];
						PxArray<AABBOverlap>	mDestroyedOverlaps[ElementType::eCOUNT];

						PxU32					mUsedSize;				// highest used value + 1
						PxU32					mNbAggregates;

	

		// PT: TODO: even in the 3.4 trunk this stuff is a clumsy mess: groups are "BpHandle" suddenly passed
		// to BroadPhaseUpdateData as "ShapeHandle".
		//Free aggregate group ids.
						PxU32					mAggregateGroupTide;
				PxArray<Bp::FilterGroup::Enum>	mFreeAggregateGroups;	// PT: TODO: remove this useless array
						PxU64					mContextID;
						bool					mOriginShifted;

		EHD void					releaseAggregateGroup(const Bp::FilterGroup::Enum group)
												{
													E_ASSERT(group != Bp::FilterGroup::eINVALID);
													mFreeAggregateGroups.pushBack(group);
												}

		EHD Bp::FilterGroup::Enum	getAggregateGroup()
												{
													PxU32 id;
													if (mFreeAggregateGroups.size())
														id = mFreeAggregateGroups.popBack();
													else
													{
														id = mAggregateGroupTide--;
														id <<= BP_FILTERING_TYPE_SHIFT_BIT;
														id |= FilterType::AGGREGATE;
													}
													const Bp::FilterGroup::Enum group = Bp::FilterGroup::Enum(id);
													E_ASSERT(group != Bp::FilterGroup::eINVALID);
													return group;
												}
	};

} //namespace Bp
} //namespace px

// ===== 원본: physx/source/lowlevelaabb/src/BpAABBManagerBase.cpp
using namespace px;
using namespace Bp;

AABBManagerBase::AABBManagerBase(	BroadPhase& bp, BoundsArray& boundsArray, PxFloatArrayPinnedSafe& contactDistance,
									PxU32 maxNbAggregates, PxU32 maxNbShapes, PxVirtualAllocator& allocator, PxU64 contextID,
									PxPairFilteringMode::Enum kineKineFilteringMode, PxPairFilteringMode::Enum staticKineFilteringMode) :
	mAddedHandleMap			(allocator.getCallback()),
	mRemovedHandleMap		(allocator.getCallback()),
	mChangedHandleMap		(allocator.getCallback()),
	mGroups					(allocator.getCallback()),
	mEnvIDs					(allocator.getCallback()),
	mContactDistance		(contactDistance),
	mVolumeData				(allocator.getCallback()),
	mFilters				(kineKineFilteringMode == PxPairFilteringMode::eKILL, staticKineFilteringMode == PxPairFilteringMode::eKILL),
	mAddedHandles			(allocator.getCallback()),
	mUpdatedHandles			(allocator.getCallback()),
	mRemovedHandles			(allocator.getCallback()),
	mBroadPhase				(bp),
	mBoundsArray			(boundsArray),
	mUsedSize				(0),
	mNbAggregates			(0),
	
	mAggregateGroupTide		(PxU32(Bp::FilterGroup::eAGGREGATE_BASE)),
	mContextID				(contextID),
	mOriginShifted			(false)
{
	E_UNUSED(maxNbAggregates);	// PT: TODO: use it or remove it
	reserveShapeSpace(PxMax(maxNbShapes, 1u));

	//	mCreatedOverlaps.reserve(16000);
}

void AABBManagerBase::reserveShapeSpace(PxU32 nbTotalBounds)
{
	nbTotalBounds = PxNextPowerOfTwo(nbTotalBounds);
	mGroups.resize(nbTotalBounds, Bp::FilterGroup::eINVALID);
	mVolumeData.resize(nbTotalBounds);					//KS - must be initialized so that userData is NULL for SQ-only shapes
	mContactDistance.resizeUninitialized(nbTotalBounds);
	mAddedHandleMap.resize(nbTotalBounds);
	mRemovedHandleMap.resize(nbTotalBounds);
}

void AABBManagerBase::reserveSpaceForBounds(BoundsIndex index)
{
	if ((index + 1) >= mVolumeData.size())
		reserveShapeSpace(index + 1);

	resetEntry(index); //KS - make sure this entry is flagged as invalid
}

void AABBManagerBase::freeBuffers()
{
	// PT: TODO: investigate if we need more stuff here
	mBroadPhase.freeBuffers();
}

void AABBManagerBase::shiftOrigin(const PxVec3& shift)
{
	mBroadPhase.shiftOrigin(shift, mBoundsArray.begin(), mContactDistance.begin());
	mOriginShifted = true;
}

// ===== 원본: physx/source/lowlevelaabb/include/BpAABBManager.h
namespace px
{
namespace Cm
{
	class FlushPool;
}

namespace Bp
{
	struct BroadPhasePair;

	class Aggregate;
	class PersistentPairs;
	class PersistentActorAggregatePair;
	class PersistentAggregateAggregatePair;
	class PersistentSelfCollisionPairs;

	struct AggPair
	{
		EHD AggPair() {}
		EHD	AggPair(ShapeHandle index0, ShapeHandle index1) : mIndex0(index0), mIndex1(index1)	{}
		ShapeHandle	mIndex0;
		ShapeHandle	mIndex1;

		EHD bool operator==(const AggPair& p) const
		{
			return (p.mIndex0 == mIndex0) && (p.mIndex1 == mIndex1);
		}
	};
	typedef PxCoalescedHashMap<AggPair, PersistentPairs*> AggPairMap;

	// PT: TODO: isn't there a generic pair structure somewhere? refactor with AggPair anyway
	struct Pair
	{
		EHD	Pair(PxU32 id0, PxU32 id1) : mID0(id0), mID1(id1)	{}
		EHD Pair(){}

			EHD bool operator<(const Pair& p) const
			{
				const PxU64 value0 = *reinterpret_cast<const PxU64*>(this);
				const PxU64 value1 = *reinterpret_cast<const PxU64*>(&p);
				return value0 < value1;
			}

			EHD bool operator==(const Pair& p) const
			{
				const PxU64 value0 = *reinterpret_cast<const PxU64*>(this);
				const PxU64 value1 = *reinterpret_cast<const PxU64*>(&p);
				return value0 == value1;
			}

			EHD bool operator!=(const Pair& p) const
			{
				const PxU64 value0 = *reinterpret_cast<const PxU64*>(this);
				const PxU64 value1 = *reinterpret_cast<const PxU64*>(&p);
				return value0 != value1;
			}

		PxU32	mID0;
		PxU32	mID1;
	};

	class AABBManager;

	class PostBroadPhaseStage2Task : public Cm::Task
	{
		Cm::FlushPool* mFlushPool;
		AABBManager& mManager;

		E_NOCOPY(PostBroadPhaseStage2Task)
	public:

		PostBroadPhaseStage2Task(PxU64 contextID, AABBManager& manager) : Cm::Task(contextID), mFlushPool(NULL), mManager(manager)
		{
		}

		virtual const char* getName() const { return "PostBroadPhaseStage2Task"; }

		void setFlushPool(Cm::FlushPool* pool) { mFlushPool = pool; }

		virtual void runInternal();
	};

	class ProcessAggPairsBase;

	/**
	\brief A structure responsible for:
	* storing an aabb representation for each active shape in the related scene
	* managing the creation/removal of aabb representations when their related shapes are created/removed
	* updating all aabbs that require an update due to modification of shape geometry or transform
	* updating the aabb of all aggregates from the union of the aabbs of all shapes that make up each aggregate
	* computing and reporting the incremental changes to the set of overlapping aabb pairs 
	*/
	class AABBManager : public AABBManagerBase
	{
													E_NOCOPY(AABBManager)
	public:
														AABBManager(BroadPhase& bp, BoundsArray& boundsArray, PxFloatArrayPinnedSafe& contactDistance,
																	PxU32 maxNbAggregates, PxU32 maxNbShapes, PxVirtualAllocator& allocator, PxU64 contextID,
																	PxPairFilteringMode::Enum kineKineFilteringMode, PxPairFilteringMode::Enum staticKineFilteringMode);

		virtual											~AABBManager() {}

		// AABBManagerBase
		virtual			void							destroy()	EPX_OVERRIDE	EPX_FINAL;
		virtual			AggregateHandle					createAggregate(BoundsIndex index, Bp::FilterGroup::Enum group, void* userData, PxU32 maxNumShapes, PxAggregateFilterHint filterHint, PxU32 envID)	EPX_OVERRIDE	EPX_FINAL;
		virtual			bool							destroyAggregate(BoundsIndex& index, Bp::FilterGroup::Enum& group, AggregateHandle aggregateHandle)	EPX_OVERRIDE	EPX_FINAL;
		virtual			bool							addBounds(BoundsIndex index, PxReal contactDistance, Bp::FilterGroup::Enum group, void* userdata, AggregateHandle aggregateHandle, ElementType::Enum volumeType, PxU32 envID)	EPX_OVERRIDE	EPX_FINAL;
		virtual			bool							removeBounds(BoundsIndex index)	EPX_OVERRIDE	EPX_FINAL;
		virtual			void							updateBPFirstPass(PxU32 numCpuTasks, Cm::FlushPool& flushPool, bool hasContactDistanceUpdated, PxBaseTask* continuation)	EPX_OVERRIDE	EPX_FINAL;
		virtual			void							updateBPSecondPass(PxcScratchAllocator* scratchAllocator, PxBaseTask* continuation)	EPX_OVERRIDE	EPX_FINAL;
		virtual			void							postBroadPhase(PxBaseTask*, Cm::FlushPool& flushPool)	EPX_OVERRIDE	EPX_FINAL;
		virtual			void							reallocateChangedAABBMgActorHandleMap(const PxU32 size)	EPX_OVERRIDE	EPX_FINAL;
		virtual			bool							getOutOfBoundsObjects(OutOfBoundsData& data)			EPX_OVERRIDE	EPX_FINAL;
		virtual			void							clearOutOfBoundsObjects()								EPX_OVERRIDE	EPX_FINAL;
		virtual			void							visualize(PxRenderOutput& out)	EPX_OVERRIDE	EPX_FINAL;
		virtual			void							releaseDeferredAggregateIds()	EPX_OVERRIDE	EPX_FINAL	{}
		//~AABBManagerBase

						void							preBpUpdate_CPU(PxU32 numCpuTasks);

						// PT: TODO: what is that BpCacheData for?
						BpCacheData*					getBpCacheData();
						void							putBpCacheData(BpCacheData*);
						void							resetBpCacheData();

						PxMutex							mMapLock;
	private:
						//void reserveShapeSpace(PxU32 nbShapes);

						void							postBpStage2(PxBaseTask*, Cm::FlushPool&);
						void							postBpStage3(PxBaseTask*);

						PostBroadPhaseStage2Task									mPostBroadPhase2;
						Cm::DelegateTask<AABBManager, &AABBManager::postBpStage3>	mPostBroadPhase3;

						PreBpUpdateTask					mPreBpUpdateTask;

						PxU32							mTimestamp;
						PxU32							mFirstFreeAggregate;
						PxArray<Aggregate*>				mAggregates;		// PT: indexed by AggregateHandle
						PxArray<Aggregate*>				mDirtyAggregates;

						AggPairMap						mActorAggregatePairs;
						AggPairMap						mAggregateAggregatePairs;

						PxArray<ProcessAggPairsBase*>	mAggPairTasks;

						PxHashSet<Pair>					mCreatedPairsTmp;	// PT: temp hashset for dubious post filtering, persistent to minimize allocs

						PxSList							mBpThreadContextPool;

						PxArray<void*>					mOutOfBoundsObjects;
						PxArray<void*>					mOutOfBoundsAggregates;

		EHD	Aggregate*						getAggregateFromHandle(AggregateHandle handle)
														{
															E_ASSERT(handle<mAggregates.size());
															return mAggregates[handle];
														}

						void							startAggregateBoundsComputationTasks(PxU32 nbToGo, PxU32 numCpuTasks, Cm::FlushPool& flushPool);
						PersistentActorAggregatePair*	createPersistentActorAggregatePair(ShapeHandle volA, ShapeHandle volB);
					PersistentAggregateAggregatePair*	createPersistentAggregateAggregatePair(ShapeHandle volA, ShapeHandle volB);
						void							updatePairs(PersistentPairs& p, BpCacheData* data = NULL);
						void							handleOriginShift();

	public:
						void							processBPCreatedPair(const BroadPhasePair& pair);
						void							processBPDeletedPair(const BroadPhasePair& pair);

		friend class PersistentActorAggregatePair;
		friend class PersistentAggregateAggregatePair;
		friend class ProcessSelfCollisionPairsParallel;
		friend class PostBroadPhaseStage2Task;
	};

} //namespace Bp
} //namespace px

// ===== 원본: physx/source/lowlevelaabb/src/BpAABBManager.cpp
#define NB_SENTINELS	6


using namespace px;
using namespace Bp;
using namespace Cm;
using namespace aos;

static const bool gSingleThreaded = false;
	#define ABP_SIMD_OVERLAP
	typedef AABB_YZn	AABB_YZ;

// (앞 조각과 같은 정의라 뺌) static const bool gUseRegularBPKernel = false;	// false to use "version 13" in box pruning series
// (앞 조각과 같은 정의라 뺌) static const bool gUnrollLoop = true;			// true to use "version 14" in box pruning series

namespace px
{
namespace Bp
{
static EHD uint32_t PxComputeHash(const Pair& p)
{
	return PxU32(px::PxComputeHash( (p.mID0&0xffff)|(p.mID1<<16)) );
}

static EHD uint32_t PxComputeHash(const AggPair& p)
{
	return PxU32(px::PxComputeHash( (p.mIndex0&0xffff)|(p.mIndex1<<16)) );
}

static EHD bool shouldPairBeDeleted(const GroupsArrayPinnedSafe& groups, ShapeHandle h0, ShapeHandle h1)
{
	E_ASSERT(h0<groups.size());
	E_ASSERT(h1<groups.size());
	return (groups[h0]==Bp::FilterGroup::eINVALID) || (groups[h1]==Bp::FilterGroup::eINVALID);
}

///

	typedef PxU32		InflatedType;

	// PT: TODO: revisit/optimize all that stuff once it works
	class Aggregate : public PxUserAllocated
	{
		public:
//														Aggregate(BoundsIndex index, bool selfCollisions);
														Aggregate(BoundsIndex index, PxAggregateFilterHint filterHint);
														~Aggregate();

						const BoundsIndex				mIndex;
		private:
						PxArray<BoundsIndex>			mAggregated;	// PT: TODO: replace with linked list?
		public:
						PersistentSelfCollisionPairs*	mSelfCollisionPairs;
						PxU32							mDirtyIndex;	// PT: index in mDirtyAggregates
		private:
						AABB_Xi*						mInflatedBoundsX;
						AABB_YZ*						mInflatedBoundsYZ;
						PxU32							mAllocatedSize;
		public:
		EHD	PxU32							getNbAggregated()		const	{ return mAggregated.size();					}
		EHD	BoundsIndex						getAggregated(PxU32 i)	const	{ return mAggregated[i];						}
		EHD	const BoundsIndex*				getIndices()			const	{ return mAggregated.begin();					}
		EHD	void							addAggregated(BoundsIndex i)	{ mAggregated.pushBack(i);						}
		EHD	bool							removeAggregated(BoundsIndex i)	{ return mAggregated.findAndReplaceWithLast(i);	}	// PT: TODO: optimize?
		EHD	const PxBounds3&				getMergedBounds()		const	{ return mBounds;								}

		EHD	void							resetDirtyState()				{ mDirtyIndex = EPX_INVALID_U32;				}
		EHD	bool							isDirty()				const	{ return mDirtyIndex != EPX_INVALID_U32;		}
		EHD void							markAsDirty(PxArray<Aggregate*>& dirtyAggregates)
														{
															if(!isDirty())
															{
																mDirtyIndex = dirtyAggregates.size();
																dirtyAggregates.pushBack(this);
															}
														}

						void							allocateBounds();
						void							computeBounds(const PxBounds3*  bounds, const float*  contactDistances) /**/;

		EHD	const AABB_Xi*					getBoundsX()	const	{ return mInflatedBoundsX;	}
		EHD	const AABB_YZ*					getBoundsYZ()	const	{ return mInflatedBoundsYZ;	}
		EHD	void							getSortedMinBounds()
														{
															if(mDirtySort)
																sortBounds();
														}
		EHD	PxAggregateFilterHint			getFilterHint()	const	{ return mFilterHint;		}
		private:
						PxBounds3						mBounds;
						PxAggregateFilterHint			mFilterHint;
						bool							mDirtySort;

						void							sortBounds();
						E_NOCOPY(Aggregate)
	};

///

namespace
{
#define MBP_ALLOC(x)		EPX_ALLOC(x, "MBP")
#define MBP_ALLOC_TMP(x)	EPX_ALLOC(x, "MBP_TMP")
#define MBP_FREE(x)			E_FREE(x)

	struct MBPEntry;
	struct RegionHandle;
	struct MBP_Object;

	class MBP_PairManager : public PairManagerData
	{
		public:
											MBP_PairManager()	{}
											~MBP_PairManager()	{}

		EHD	InternalPair*		addPair(PxU32 id0, PxU32 id1);
	};

///////////////////////////////////////////////////////////////////////////////

EHD InternalPair* MBP_PairManager::addPair(PxU32 id0, PxU32 id1)
{
	E_ASSERT(id0!=INVALID_ID);
	E_ASSERT(id1!=INVALID_ID);
	return addPairInternal(id0, id1);
}

///////////////////////////////////////////////////////////////////////////////

}

	typedef MBP_PairManager	PairArray;

///

class PersistentPairs : public PxUserAllocated
{
	public:
									PersistentPairs() : mTimestamp(EPX_INVALID_U32), mShouldBeDeleted(false)	{}
	virtual							~PersistentPairs()														{}

	virtual			bool			update(AABBManager& /*manager*/, BpCacheData* /*data*/ = NULL) { return false; }


	EHD	void			updatePairs(PxU32 timestamp, const PxBounds3* bounds, const float* contactDistances, const Bp::FilterGroup::Enum* groups,
												const bool* lut, VolumeData* volumeData, PxArray<AABBOverlap>* createdOverlaps, PxArray<AABBOverlap>* destroyedOverlaps);
					void			outputDeletedOverlaps(PxArray<AABBOverlap>* overlaps, const VolumeData* volumeData);
	private:
	virtual			void			findOverlaps(PairArray& pairs, const PxBounds3*  bounds, const float*  contactDistances, const Bp::FilterGroup::Enum*  groups, const bool*  lut)	= 0;
	protected:
					PxU32			mTimestamp;
					MBP_PairManager	mPM;
	public:
					bool			mShouldBeDeleted;
};

/////
	#define PosXType2	PxU32

	#define SIMD_OVERLAP_TEST_14a(box)	em_movemask_ps(em_cmpngt_ps(b, em_load_ps(box)))==15

	#define SIMD_OVERLAP_INIT_9c(box)	\
			em128 b = em_shuffle_ps(em_load_ps(&box.mMinY), em_load_ps(&box.mMinY), 78);\
			const float Coeff = -1.0f;\
			b = em_mul_ps(b, em_load1_ps(&Coeff));

	#define SIMD_OVERLAP_TEST_9c(box)					\
			const em128 a = em_load_ps(&box.mMinY);	\
			const em128 d = em_cmpge_ps(a, b);		\
			if(em_movemask_ps(d)==15)

	#define	CODEALIGN16		//_asm	align 16
	#define SIMD_OVERLAP_PRELOAD_BOX0	SIMD_OVERLAP_INIT_9c(box0)
	#define SIMD_OVERLAP_TEST(x)		SIMD_OVERLAP_TEST_9c(x)


	#define ABP_OVERLAP_TEST(x)	SIMD_OVERLAP_TEST(x)

//#define BIP_VERSION_1

	struct outputPair_Bipartite
	{
		outputPair_Bipartite(PairArray* pairManager, const BoundsIndex* remap0, const BoundsIndex* remap1, const Bp::FilterGroup::Enum* groups, const bool* lut) :
			mPairManager	(pairManager),
			mRemap0			(remap0),
			mRemap1			(remap1),
			mGroups			(groups),
			mLUT			(lut)
		{
		}

		EHD	void outputPair(PxU32 index0, PxU32 index1)
		{
			const PxU32 aggIndex0 = mRemap0[index0];
			const PxU32 aggIndex1 = mRemap1[index1];
			if(groupFiltering(mGroups[aggIndex0], mGroups[aggIndex1], mLUT))
				mPairManager->addPair(aggIndex0, aggIndex1);
		}

		PairArray*						mPairManager;
		const BoundsIndex*				mRemap0;
		const BoundsIndex*				mRemap1;
		const Bp::FilterGroup::Enum*	mGroups;
		const bool*						mLUT;
	};

template<int codepath>
static void boxPruningKernel(	PairArray*  pairManager, const bool*  lut,
								PxU32 nb0, const BoundsIndex*  remap0, const AABB_Xi*  boxes0_X, const AABB_YZ*  boxes0_YZ,
								PxU32 nb1, const BoundsIndex*  remap1, const AABB_Xi*  boxes1_X, const AABB_YZ*  boxes1_YZ,
								const Bp::FilterGroup::Enum*  groups)
{
//	outputPair_Bipartite pm(pairManager, aggregate0->getIndices(), aggregate1->getIndices(), groups, lut);
	outputPair_Bipartite pm(pairManager, remap0, remap1, groups, lut);

	PxU32 index0 = 0;
	PxU32 runningIndex1 = 0;

	while(runningIndex1<nb1 && index0<nb0)
	{
		const AABB_Xi& box0_X = boxes0_X[index0];
		const PosXType2 maxLimit = box0_X.mMaxX;

		const PosXType2 minLimit = box0_X.mMinX;
		if(!codepath)
		{
			while(boxes1_X[runningIndex1].mMinX<minLimit)
				runningIndex1++;
		}
		else
		{
			while(boxes1_X[runningIndex1].mMinX<=minLimit)
				runningIndex1++;
		}

		const AABB_YZ& box0 = boxes0_YZ[index0];
		SIMD_OVERLAP_PRELOAD_BOX0

		if(gUseRegularBPKernel)
		{
			PxU32 index1 = runningIndex1;

			while(boxes1_X[index1].mMinX<=maxLimit)
			{
				ABP_OVERLAP_TEST(boxes1_YZ[index1])
				{
					pm.outputPair(index0, index1);
				}
				index1++;
			}
		}
		else
		{
			PxU32 Offset = 0;
			const char* const CurrentBoxListYZ = reinterpret_cast<const char*>(&boxes1_YZ[runningIndex1]);
			const char* const CurrentBoxListX = reinterpret_cast<const char*>(&boxes1_X[runningIndex1]);

			if(!gUnrollLoop)
			{
				while(*reinterpret_cast<const PosXType2*>(CurrentBoxListX + Offset)<=maxLimit)
				{
					const float* box = reinterpret_cast<const float*>(CurrentBoxListYZ + Offset*2);
					if(SIMD_OVERLAP_TEST_14a(box))
					{
						const PxU32 Index1 = PxU32(CurrentBoxListX + Offset - reinterpret_cast<const char*>(boxes1_X))>>3;
						pm.outputPair(index0, Index1);
					}
					Offset += 8;
				}
			}
			else
			{

#define BIP_VERSION4
	#define BLOCK4(x, label)	{const float* box = reinterpret_cast<const float*>(CurrentBoxListYZ + Offset*2 + x*2);	\
								if(SIMD_OVERLAP_TEST_14a(box))	\
								goto label;	}
		goto StartLoop4;
		CODEALIGN16
FoundOverlap3:
		Offset += 8;
		CODEALIGN16
FoundOverlap2:
		Offset += 8;
		CODEALIGN16
FoundOverlap1:
		Offset += 8;
		CODEALIGN16
FoundOverlap0:
		Offset += 8;
		CODEALIGN16
FoundOverlap:
		{
			const PxU32 Index1 = PxU32(CurrentBoxListX + Offset - 8 - reinterpret_cast<const char*>(boxes1_X))>>3;
			pm.outputPair(index0, Index1);
		}
		CODEALIGN16
StartLoop4:
		while(*reinterpret_cast<const PosXType2*>(CurrentBoxListX + Offset + 8*5)<=maxLimit)
		{
			BLOCK4(0, FoundOverlap0)
			BLOCK4(8, FoundOverlap1)
			BLOCK4(16, FoundOverlap2)
			BLOCK4(24, FoundOverlap3)
			Offset += 40;
			BLOCK4(-8, FoundOverlap)
		}
#undef BLOCK4

	#define BLOCK	if(*reinterpret_cast<const PosXType2*>(CurrentBoxListX + Offset)<=maxLimit)			\
				{if(SIMD_OVERLAP_TEST_14a(reinterpret_cast<const float*>(CurrentBoxListYZ + Offset*2)))	\
						goto OverlapFound;										\
					Offset += 8;

		goto LoopStart;
		CODEALIGN16
OverlapFound:
		{
			const PxU32 Index1 = PxU32(CurrentBoxListX + Offset - reinterpret_cast<const char*>(boxes1_X))>>3;
			pm.outputPair(index0, Index1);
		}
		Offset += 8;
		CODEALIGN16
LoopStart:
		BLOCK
			BLOCK
				BLOCK
				}
			}
			goto LoopStart;
		}
#undef BLOCK
			}
		}

		index0++;
	}
}

static EHD void doBipartiteBoxPruning_Leaf(
		PairArray*  pairManager, const bool*  lut,
		Aggregate*  aggregate0, Aggregate*  aggregate1, const Bp::FilterGroup::Enum*  groups)
{
	const PxU32 nb0 =  aggregate0->getNbAggregated();
	const PxU32 nb1 =  aggregate1->getNbAggregated();
	const BoundsIndex*  remap0 = aggregate0->getIndices();
	const BoundsIndex*  remap1 = aggregate1->getIndices();
	const AABB_Xi*  boxes0_X = aggregate0->getBoundsX();
	const AABB_YZ*  boxes0_YZ = aggregate0->getBoundsYZ();
	const AABB_Xi*  boxes1_X = aggregate1->getBoundsX();
	const AABB_YZ*  boxes1_YZ = aggregate1->getBoundsYZ();
	boxPruningKernel<0>(pairManager, lut, nb0, remap0, boxes0_X, boxes0_YZ, nb1, remap1, boxes1_X, boxes1_YZ, groups);
	boxPruningKernel<1>(pairManager, lut, nb1, remap1, boxes1_X, boxes1_YZ, nb0, remap0, boxes0_X, boxes0_YZ, groups);
}

	struct outputPair_Complete
	{
		outputPair_Complete(PairArray* pairManager, Aggregate* aggregate, const Bp::FilterGroup::Enum* groups, const bool* lut) :
			mPairManager	(pairManager),
			mAggregate		(aggregate),
			mGroups			(groups),
			mLUT			(lut)
		{
		}

		EHD	void outputPair(PxU32 index0, PxU32 index1)
		{
			const PxU32 aggIndex0 = mAggregate->getAggregated(index0);
			const PxU32 aggIndex1 = mAggregate->getAggregated(index1);

			if(groupFiltering(mGroups[aggIndex0], mGroups[aggIndex1], mLUT))
				mPairManager->addPair(aggIndex0, aggIndex1);
		}

		PairArray*						mPairManager;
		Aggregate*						mAggregate;
		const Bp::FilterGroup::Enum*	mGroups;
		const bool*						mLUT;
	};

static void doCompleteBoxPruning_Leaf(	PairArray*  pairManager, const bool*  lut,
										Aggregate*  aggregate, const Bp::FilterGroup::Enum*  groups)
{
	outputPair_Complete pm(pairManager, aggregate, groups, lut);
	const PxU32 nb =  aggregate->getNbAggregated();
	const AABB_Xi*  boxes_X = aggregate->getBoundsX();
	const AABB_YZ*  boxes_YZ = aggregate->getBoundsYZ();

	PxU32 index0 = 0;
	PxU32 runningIndex = 0;
	while(runningIndex<nb && index0<nb)
	{
		const AABB_Xi& box0_X = boxes_X[index0];
		const PosXType2 maxLimit = box0_X.mMaxX;

		const PosXType2 minLimit = box0_X.mMinX;
		while(boxes_X[runningIndex++].mMinX<minLimit);

		const AABB_YZ& box0 = boxes_YZ[index0];
		SIMD_OVERLAP_PRELOAD_BOX0

		if(gUseRegularBPKernel)
		{
			PxU32 index1 = runningIndex;
			while(boxes_X[index1].mMinX<=maxLimit)
			{
				ABP_OVERLAP_TEST(boxes_YZ[index1])
				{
					pm.outputPair(index0, index1);
				}
				index1++;
			}
		}
		else
		{
			PxU32 Offset = 0;
			const char* const CurrentBoxListYZ = reinterpret_cast<const char*>(&boxes_YZ[runningIndex]);
			const char* const CurrentBoxListX = reinterpret_cast<const char*>(&boxes_X[runningIndex]);

			if(!gUnrollLoop)
			{
				while(*reinterpret_cast<const PosXType2*>(CurrentBoxListX + Offset)<=maxLimit)
				{
					const float* box = reinterpret_cast<const float*>(CurrentBoxListYZ + Offset*2);
					if(SIMD_OVERLAP_TEST_14a(box))
					{
						const PxU32 Index = PxU32(CurrentBoxListX + Offset - reinterpret_cast<const char*>(boxes_X))>>3;
						pm.outputPair(index0, Index);
					}
					Offset += 8;
				}
			}
			else
			{

#define VERSION4c
#define VERSION3	// Enable this as our safe loop
	#define BLOCK4(x, label)	{const float* box = reinterpret_cast<const float*>(CurrentBoxListYZ + Offset*2 + x*2);	\
							if(SIMD_OVERLAP_TEST_14a(box))					\
								goto label;	}
		goto StartLoop4;
		CODEALIGN16
FoundOverlap3:
		Offset += 8;
		CODEALIGN16
FoundOverlap2:
		Offset += 8;
		CODEALIGN16
FoundOverlap1:
		Offset += 8;
		CODEALIGN16
FoundOverlap0:
		Offset += 8;
		CODEALIGN16
FoundOverlap:
		{
			const PxU32 Index = PxU32(CurrentBoxListX + Offset - 8 - reinterpret_cast<const char*>(boxes_X))>>3;
			pm.outputPair(index0, Index);
		}
		CODEALIGN16
StartLoop4:
		while(*reinterpret_cast<const PosXType2*>(CurrentBoxListX + Offset + 8*5)<=maxLimit)
		{
			BLOCK4(0, FoundOverlap0)
			BLOCK4(8, FoundOverlap1)
			BLOCK4(16, FoundOverlap2)
			BLOCK4(24, FoundOverlap3)
			Offset += 40;
			BLOCK4(-8, FoundOverlap)
		}

#define VERSION3
	#define BLOCK	if(*reinterpret_cast<const PosXType2*>(CurrentBoxListX + Offset)<=maxLimit)			\
				{if(SIMD_OVERLAP_TEST_14a(reinterpret_cast<const float*>(CurrentBoxListYZ + Offset*2)))	\
						goto BeforeLoop;										\
					Offset += 8;

		goto StartLoop;
		CODEALIGN16
BeforeLoop:
		{
			const PxU32 Index = PxU32(CurrentBoxListX + Offset - reinterpret_cast<const char*>(boxes_X))>>3;
			pm.outputPair(index0, Index);
			Offset += 8;
		}
		CODEALIGN16
StartLoop:
		BLOCK
			BLOCK
				BLOCK
					BLOCK
						BLOCK
						}
					}
				}
			}
			goto StartLoop;
		}
			}
		}

		index0++;
	}
}

/////

class PersistentActorAggregatePair : public PersistentPairs
{
	public:
								PersistentActorAggregatePair(Aggregate* aggregate, ShapeHandle actorHandle);
	virtual						~PersistentActorAggregatePair()	{}

	virtual			void		findOverlaps(PairArray& pairs, const PxBounds3*  bounds, const float*  contactDistances, const Bp::FilterGroup::Enum*  groups, const bool*  lut);
	virtual			bool		update(AABBManager& manager, BpCacheData* data);

					ShapeHandle	mAggregateHandle;
					ShapeHandle	mActorHandle;
					Aggregate*	mAggregate;
};

PersistentActorAggregatePair::PersistentActorAggregatePair(Aggregate* aggregate, ShapeHandle actorHandle) :
	mAggregateHandle	(aggregate->mIndex),
	mActorHandle		(actorHandle),
	mAggregate			(aggregate)
{
}

void PersistentActorAggregatePair::findOverlaps(PairArray& pairs, const PxBounds3*  bounds, const float*  contactDistances, const Bp::FilterGroup::Enum*  groups, const bool*  lut)
{
	if(0)
	{
		Aggregate singleActor(INVALID_ID, false);
		singleActor.addAggregated(mActorHandle);
		singleActor.allocateBounds();
		singleActor.computeBounds(bounds, contactDistances);
		singleActor.getSortedMinBounds();
		mAggregate->getSortedMinBounds();
		doBipartiteBoxPruning_Leaf(&pairs, lut, &singleActor, mAggregate, groups);
	}
	else
	{

	mAggregate->getSortedMinBounds();
	const PxU32 nb0 = mAggregate->getNbAggregated();
	const BoundsIndex*  remap0 = mAggregate->getIndices();
	const AABB_Xi*  boxes0_X = mAggregate->getBoundsX();
	const AABB_YZ*  boxes0_YZ = mAggregate->getBoundsYZ();

	alignas(16) AABB_Xi inflatedBoundsX[1+NB_SENTINELS];
	alignas(16) AABB_YZ inflatedBoundsYZ;

	// Compute bounds
	{
		PxVec4 boxMin;
		PxVec4 boxMax;
		const BoundsIndex actorHandle = mActorHandle;
		const PxBounds3& b = bounds[actorHandle];
		const Vec4V offsetV = V4Load(contactDistances[actorHandle]);
		const Vec4V minimumV = V4Sub(V4LoadU(&b.minimum.x), offsetV);
		const Vec4V maximumV = V4Add(V4LoadU(&b.maximum.x), offsetV);
		V4StoreA(minimumV, &boxMin.x);
		V4StoreA(maximumV, &boxMax.x);
		inflatedBoundsX[0].initFromPxVec4(boxMin, boxMax);
		inflatedBoundsYZ.initFromPxVec4(boxMin, boxMax);

		for(PxU32 i=0;i<NB_SENTINELS;i++)
			inflatedBoundsX[1+i].initSentinel();
	}

	const PxU32 nb1 = 1;
	const BoundsIndex*  remap1 = &mActorHandle;
	const AABB_Xi*  boxes1_X = inflatedBoundsX;
	const AABB_YZ*  boxes1_YZ = &inflatedBoundsYZ;

	boxPruningKernel<0>(&pairs, lut, nb0, remap0, boxes0_X, boxes0_YZ, nb1, remap1, boxes1_X, boxes1_YZ, groups);
	boxPruningKernel<1>(&pairs, lut, nb1, remap1, boxes1_X, boxes1_YZ, nb0, remap0, boxes0_X, boxes0_YZ, groups);

	}
}

bool PersistentActorAggregatePair::update(AABBManager& manager, BpCacheData* data)
{
	if(mShouldBeDeleted || shouldPairBeDeleted(manager.mGroups, mAggregateHandle, mActorHandle))
		return true;

	if(!mAggregate->getNbAggregated())	// PT: needed with lazy empty actors
		return true;

	if(mAggregate->isDirty() || manager.mChangedHandleMap.boundedTest(mActorHandle))
		manager.updatePairs(*this, data);

	return false;
}

/////

class PersistentAggregateAggregatePair : public PersistentPairs
{
	public:
								PersistentAggregateAggregatePair(Aggregate* aggregate0, Aggregate* aggregate1);
	virtual						~PersistentAggregateAggregatePair()	{}

	virtual			void		findOverlaps(PairArray& pairs, const PxBounds3*  bounds, const float*  contactDistances, const Bp::FilterGroup::Enum*  groups, const bool*  lut);
	virtual			bool		update(AABBManager& manager, BpCacheData*);

					ShapeHandle	mAggregateHandle0;
					ShapeHandle	mAggregateHandle1;
					Aggregate*	mAggregate0;
					Aggregate*	mAggregate1;
};

PersistentAggregateAggregatePair::PersistentAggregateAggregatePair(Aggregate* aggregate0, Aggregate* aggregate1) :
	mAggregateHandle0	(aggregate0->mIndex),
	mAggregateHandle1	(aggregate1->mIndex),
	mAggregate0			(aggregate0),
	mAggregate1			(aggregate1)
{
}

void PersistentAggregateAggregatePair::findOverlaps(PairArray& pairs, const PxBounds3*  /*bounds*/, const float*  /*contactDistances*/, const Bp::FilterGroup::Enum*  groups, const bool*  lut)
{
	mAggregate0->getSortedMinBounds();
	mAggregate1->getSortedMinBounds();
	doBipartiteBoxPruning_Leaf(&pairs, lut, mAggregate0, mAggregate1, groups);
}

bool PersistentAggregateAggregatePair::update(AABBManager& manager, BpCacheData* data)
{
	if(mShouldBeDeleted || shouldPairBeDeleted(manager.mGroups, mAggregateHandle0, mAggregateHandle1))
		return true;

	if(!mAggregate0->getNbAggregated() || !mAggregate1->getNbAggregated())	// PT: needed with lazy empty actors
		return true;

	if(mAggregate0->isDirty() || mAggregate1->isDirty())
		manager.updatePairs(*this, data);

	return false;
}

/////

class PersistentSelfCollisionPairs : public PersistentPairs
{
	public:
						PersistentSelfCollisionPairs(Aggregate* aggregate);
	virtual				~PersistentSelfCollisionPairs()	{}

	virtual	void		findOverlaps(PairArray& pairs, const PxBounds3*  bounds, const float*  contactDistances, const Bp::FilterGroup::Enum*  groups, const bool*  lut);

			Aggregate*	mAggregate;
};

PersistentSelfCollisionPairs::PersistentSelfCollisionPairs(Aggregate* aggregate) :
	mAggregate	(aggregate)
{
}

void PersistentSelfCollisionPairs::findOverlaps(PairArray& pairs, const PxBounds3* /*bounds*/, const float* /*contactDistances*/, const Bp::FilterGroup::Enum*  groups, const bool*  lut)
{
	mAggregate->getSortedMinBounds();
	doCompleteBoxPruning_Leaf(&pairs, lut, mAggregate, groups);
}

/////

Aggregate::Aggregate(BoundsIndex index, PxAggregateFilterHint filterHint) :
	mIndex				(index),
	mInflatedBoundsX	(NULL),
	mInflatedBoundsYZ	(NULL),
	mAllocatedSize		(0),
	mFilterHint			(filterHint),
	mDirtySort			(false)
{
	resetDirtyState();
	const PxU32 selfCollisions = PxGetAggregateSelfCollisionBit(filterHint);
	mSelfCollisionPairs = selfCollisions ? EPX_NEW(PersistentSelfCollisionPairs)(this) : NULL;
}

Aggregate::~Aggregate()
{
	E_FREE(mInflatedBoundsYZ);
	E_FREE(mInflatedBoundsX);

	E_DELETE(mSelfCollisionPairs);
}

void Aggregate::sortBounds()
{
	mDirtySort = false;
	const PxU32 nbObjects = getNbAggregated();
	if(nbObjects<2)
		return;

	{
		EPX_ALLOCA(minPosBounds, InflatedType, nbObjects+1);
		bool alreadySorted =  true;
		InflatedType previousB = mInflatedBoundsX[0].mMinX;
		minPosBounds[0] = previousB;
		for(PxU32 i=1;i<nbObjects;i++)
		{
			const InflatedType minB = mInflatedBoundsX[i].mMinX;
			if(minB<previousB)
				alreadySorted = false;
			previousB = minB;
			minPosBounds[i] = minB;
		}
		if(alreadySorted)
			return;

		{
		Cm::RadixSortBuffered mRS;

		minPosBounds[nbObjects] = 0xffffffff;
		mRS.Sort(minPosBounds, nbObjects+1, /*RadixHint::*/RADIX_UNSIGNED);

		if(0)
		{
			PxArray<PxU32> copy = mAggregated;
			AABB_Xi* boundsXCopy = EPX_ALLOCATE(AABB_Xi, nbObjects, "mInflatedBounds");
			AABB_YZ* boundsYZCopy = EPX_ALLOCATE(AABB_YZ, nbObjects, "mInflatedBounds");
			PxMemCopy(boundsXCopy, mInflatedBoundsX, nbObjects*sizeof(AABB_Xi));
			PxMemCopy(boundsYZCopy, mInflatedBoundsYZ, nbObjects*sizeof(AABB_YZ));
			const PxU32* Sorted = mRS.GetRanks();
			for(PxU32 i=0;i<nbObjects;i++)
			{
				const PxU32 sortedIndex = Sorted[i];
				mAggregated[i] = copy[sortedIndex];
				mInflatedBoundsX[i] = boundsXCopy[sortedIndex];
				mInflatedBoundsYZ[i] = boundsYZCopy[sortedIndex];
			}
			E_FREE(boundsYZCopy);
			E_FREE(boundsXCopy);
		}
		else
		{
			PxArray<PxU32> copy = mAggregated;	// PT: TODO: revisit this, avoid the copy like we do for the other buffers
			AABB_Xi* sortedBoundsX = EPX_ALLOCATE(AABB_Xi, (nbObjects+NB_SENTINELS), "mInflatedBounds");
			AABB_YZ* sortedBoundsYZ = EPX_ALLOCATE(AABB_YZ, (nbObjects), "mInflatedBounds");

			const PxU32* Sorted = mRS.GetRanks();
			for(PxU32 i=0;i<nbObjects;i++)
			{
				const PxU32 sortedIndex = Sorted[i];
				mAggregated[i] = copy[sortedIndex];
				sortedBoundsX[i] = mInflatedBoundsX[sortedIndex];
				sortedBoundsYZ[i] = mInflatedBoundsYZ[sortedIndex];
			}
			for(PxU32 i=0;i<NB_SENTINELS;i++)
				sortedBoundsX[nbObjects+i].initSentinel();
			mAllocatedSize = nbObjects;
			E_FREE(mInflatedBoundsYZ);
			E_FREE(mInflatedBoundsX);
			mInflatedBoundsX = sortedBoundsX;
			mInflatedBoundsYZ = sortedBoundsYZ;
		}
		}
	}
}

void Aggregate::allocateBounds()
{
	const PxU32 size = getNbAggregated();
	if(size!=mAllocatedSize)
	{
		mAllocatedSize = size;
		E_FREE(mInflatedBoundsYZ);
		E_FREE(mInflatedBoundsX);
		mInflatedBoundsX = EPX_ALLOCATE(AABB_Xi, (size+NB_SENTINELS), "mInflatedBounds");
		mInflatedBoundsYZ = EPX_ALLOCATE(AABB_YZ, (size), "mInflatedBounds");
	}
}

void Aggregate::computeBounds(const PxBounds3*  bounds, const float*  contactDistances) /**/
{
//	EPX_PROFILE_ZONE("Aggregate::computeBounds",0);

	const PxU32 size = getNbAggregated();
	E_ASSERT(size);

	// PT: TODO: delay the conversion to integers until we sort (i.e. really need) the aggregated bounds?
	PxVec4 boxMin;
	PxVec4 boxMax;

	const PxU32 lookAhead = 4;
	Vec4V minimumV;
	Vec4V maximumV;
	{
		const BoundsIndex index0 = getAggregated(0);
		const PxU32 last = PxMin(lookAhead, size-1);
		for(PxU32 i=1;i<=last;i++)
		{
			const BoundsIndex index = getAggregated(i);
			PxPrefetchLine(bounds + index, 0);
			PxPrefetchLine(contactDistances + index, 0);
		}
		const PxBounds3& b = bounds[index0];
		const Vec4V offsetV = V4Load(contactDistances[index0]);
		minimumV = V4Sub(V4LoadU(&b.minimum.x), offsetV);
		maximumV = V4Add(V4LoadU(&b.maximum.x), offsetV);
		V4StoreA(minimumV, &boxMin.x);
		V4StoreA(maximumV, &boxMax.x);
		mInflatedBoundsX[0].initFromPxVec4(boxMin, boxMax);
		mInflatedBoundsYZ[0].initFromPxVec4(boxMin, boxMax);
	}

	for(PxU32 i=1;i<size;i++)
	{
		const BoundsIndex index = getAggregated(i);
		if(i+lookAhead<size)
		{
			const BoundsIndex nextIndex = getAggregated(i+lookAhead);
			PxPrefetchLine(bounds + nextIndex, 0);
			PxPrefetchLine(contactDistances + nextIndex, 0);
		}
		const PxBounds3& b = bounds[index];
		const Vec4V offsetV = V4Load(contactDistances[index]);
		const Vec4V aggregatedBoundsMinV = V4Sub(V4LoadU(&b.minimum.x), offsetV);
		const Vec4V aggregatedBoundsMaxV = V4Add(V4LoadU(&b.maximum.x), offsetV);
		minimumV = V4Min(minimumV, aggregatedBoundsMinV);
		maximumV = V4Max(maximumV, aggregatedBoundsMaxV);
		V4StoreA(aggregatedBoundsMinV, &boxMin.x);
		V4StoreA(aggregatedBoundsMaxV, &boxMax.x);
		mInflatedBoundsX[i].initFromPxVec4(boxMin, boxMax);
		mInflatedBoundsYZ[i].initFromPxVec4(boxMin, boxMax);
	}

	StoreBounds(mBounds, minimumV, maximumV);
//	StoreBounds(boundsArray.begin()[mIndex], minimumV, maximumV);
//	boundsArray.setChangedState();

/*	if(0)
	{
		const PxBounds3& previousBounds = boundsArray.getBounds(mIndex);
		if(previousBounds.minimum==aggregateBounds.minimum && previousBounds.maximum==aggregateBounds.maximum)
		{
			// PT: same bounds as before
			printf("SAME BOUNDS\n");
		}
	}*/
	for(PxU32 i=0;i<NB_SENTINELS;i++)
		mInflatedBoundsX[size+i].initSentinel();
	mDirtySort = true;
}

/////

static void buildFreeBitmap(PxBitMap& bitmap, PxU32 currentFree, const PxArray<Aggregate*>& aggregates)
{
	const PxU32 N = aggregates.size();

	bitmap.resizeAndClear(N);

	while(currentFree!=EPX_INVALID_U32)
	{
		bitmap.set(currentFree);
		currentFree = PxU32(size_t(aggregates[currentFree]));
	}
}


AABBManager::AABBManager(	BroadPhase& bp, BoundsArray& boundsArray, PxFloatArrayPinnedSafe& contactDistance,
							PxU32 maxNbAggregates, PxU32 maxNbShapes, PxVirtualAllocator& allocator, PxU64 contextID,
							PxPairFilteringMode::Enum kineKineFilteringMode, PxPairFilteringMode::Enum staticKineFilteringMode) :
	AABBManagerBase			(bp, boundsArray, contactDistance, maxNbAggregates, maxNbShapes, allocator, contextID, kineKineFilteringMode, staticKineFilteringMode),
	mPostBroadPhase2		(contextID, *this),
	mPostBroadPhase3		(contextID, this, "AABBManager::postBroadPhaseStage3"),
	mPreBpUpdateTask		(contextID),
	mTimestamp				(0),
	mFirstFreeAggregate		(EPX_INVALID_U32),
	mOutOfBoundsObjects		("AABBManager::mOutOfBoundsObjects"),
	mOutOfBoundsAggregates	("AABBManager::mOutOfBoundsAggregates")
{
}

static void releasePairs(AggPairMap& map)
{
	for(AggPairMap::Iterator iter = map.getIterator(); !iter.done(); ++iter)
		E_DELETE(iter->second);
}

void AABBManager::destroy()
{
	releasePairs(mActorAggregatePairs);
	releasePairs(mAggregateAggregatePairs);

	{
		PxBitMap bitmap;
		buildFreeBitmap(bitmap, mFirstFreeAggregate, mAggregates);

		const PxU32 nb = mAggregates.size();
		for(PxU32 i=0;i<nb;i++)
		{
			if(bitmap.test(i))
				continue;

			Aggregate* a = mAggregates[i];
			E_DELETE(a);
		}
	}

	BpCacheData* entry = static_cast<BpCacheData*>(mBpThreadContextPool.pop());
	while (entry)
	{
		entry->~BpCacheData();
		E_FREE(entry);
		entry = static_cast<BpCacheData*>(mBpThreadContextPool.pop());
	}

	EPX_DELETE_THIS;
}

static void removeAggregateFromDirtyArray(Aggregate* aggregate, PxArray<Aggregate*>& dirtyAggregates)
{
	// PT: TODO: do this lazily like for interactions?
	if(aggregate->isDirty())
	{
		const PxU32 dirtyIndex = aggregate->mDirtyIndex;
		E_ASSERT(dirtyAggregates[dirtyIndex]==aggregate);
		dirtyAggregates.replaceWithLast(dirtyIndex);
		if(dirtyIndex<dirtyAggregates.size())
			dirtyAggregates[dirtyIndex]->mDirtyIndex = dirtyIndex;
		aggregate->resetDirtyState();
	}
	else
	{
		E_ASSERT(!dirtyAggregates.findAndReplaceWithLast(aggregate));
	}
}

// PT: userData = Sc::ElementSim
bool AABBManager::addBounds(BoundsIndex index, PxReal contactDistance, Bp::FilterGroup::Enum group, void* userData, AggregateHandle aggregateHandle, ElementType::Enum volumeType, PxU32 envID)
{
	if(envID!=EPX_INVALID_U32)
	{
		envID = EPX_INVALID_U32;
		PxGetFoundation().error(PxErrorCode::eINVALID_PARAMETER, EPX_FL, "AABBManager::addBounds - environment ID is not supported in CPU broadphases\n");
	}

//	E_ASSERT(checkID(index));

	initEntry(index, contactDistance, group, userData, volumeType);

	if(aggregateHandle==EPX_INVALID_U32)
	{
		mVolumeData[index].setSingleActor();

		addBPEntry(index);
	}
	else
	{
		mVolumeData[index].setAggregated(aggregateHandle);

		Aggregate* aggregate = getAggregateFromHandle(aggregateHandle);

		{
			// PT: schedule the aggregate for BP insertion here, if we just added its first shape
			if(!aggregate->getNbAggregated())
				addBPEntry(aggregate->mIndex);

			aggregate->addAggregated(index);

			// PT: new actor added to aggregate => mark dirty to recompute bounds later
			aggregate->markAsDirty(mDirtyAggregates);
		}
	}

	return true;
}

bool AABBManager::removeBounds(BoundsIndex index)
{
	// PT: TODO: shouldn't it be compared to mUsedSize?
	E_ASSERT(index < mVolumeData.size());

	bool res = false;
	if(mVolumeData[index].isSingleActor())
	{
		res = removeBPEntry(index);
	}
	else
	{
		E_ASSERT(mVolumeData[index].isAggregated());

		const AggregateHandle aggregateHandle = mVolumeData[index].getAggregateOwner();
		Aggregate* aggregate = getAggregateFromHandle(aggregateHandle);
		bool status = aggregate->removeAggregated(index);
		(void)status;
//		E_ASSERT(status);	// PT: can be false when >128 shapes

		// PT: remove empty aggregates, otherwise the BP will crash with empty bounds
		if(!aggregate->getNbAggregated())
		{
			removeBPEntry(aggregate->mIndex);
			removeAggregateFromDirtyArray(aggregate, mDirtyAggregates);
		}
		else
			aggregate->markAsDirty(mDirtyAggregates);	// PT: actor removed from aggregate => mark dirty to recompute bounds later
	}

	resetEntry(index);
	return res;
}

// PT: TODO: the userData is actually a PxAggregate pointer. Maybe we could expose/use that.
AggregateHandle AABBManager::createAggregate(BoundsIndex index, Bp::FilterGroup::Enum group, void* userData, PxU32 /*maxNumShapes*/, PxAggregateFilterHint filterHint, PxU32 envID)
{
//	E_ASSERT(checkID(index));

	if(envID!=EPX_INVALID_U32)
	{
		envID = EPX_INVALID_U32;
		PxGetFoundation().error(PxErrorCode::eINVALID_PARAMETER, EPX_FL, "AABBManager::createAggregate - environment ID is not supported in CPU broadphases\n");
	}

	Aggregate* aggregate = EPX_NEW(Aggregate)(index, filterHint);

	AggregateHandle handle;
	if(mFirstFreeAggregate==EPX_INVALID_U32)
	{
		handle = mAggregates.size();
		mAggregates.pushBack(aggregate);
	}
	else
	{
		handle = mFirstFreeAggregate;
		mFirstFreeAggregate = PxU32(size_t(mAggregates[mFirstFreeAggregate]));
		mAggregates[handle] = aggregate;
	}

/*		PxU32 id = index;
		id<<=2;
		id|=FilterType::AGGREGATE;
		initEntry(index, 0.0f, Bp::FilterGroup::Enum(id), userData);
*/
	initEntry(index, 0.0f, getAggregateGroup(), userData);
	E_UNUSED(group);

	mVolumeData[index].setAggregate(handle);

	//creates an extra empty bound explicitly. Corresponding entry in the transform cache is guaranteed by resizing and advancing the index for next transforms, although explicitly setTransform is not called
	//bounds and transforms are in sync and this change is reflected in changes array for GPU transfer
	mBoundsArray.setBounds(PxBounds3::empty(), index);

	mNbAggregates++;

	// PT: we don't add empty aggregates to mAddedHandleMap yet, since they make the BP crash.
	return handle;
}

bool AABBManager::destroyAggregate(BoundsIndex& index_, Bp::FilterGroup::Enum& group_, AggregateHandle aggregateHandle)
{

	Aggregate* aggregate = getAggregateFromHandle(aggregateHandle);


	const BoundsIndex index = aggregate->mIndex;
	removeAggregateFromDirtyArray(aggregate, mDirtyAggregates);

	if(mAddedHandleMap.test(index))			// PT: if object had been added this frame...
		mAddedHandleMap.reset(index);		// PT: ...then simply revert the previous operation locally (it hasn't been passed to the BP yet).
	else if(aggregate->getNbAggregated())	// PT: else we need to remove it from the BP if it has been added there. If there's no aggregated
		mRemovedHandleMap.set(index);		// PT: shapes then the aggregate has never been added, or already removed.

	E_DELETE(aggregate);
	mAggregates[aggregateHandle] = reinterpret_cast<Aggregate*>(size_t(mFirstFreeAggregate));
	mFirstFreeAggregate = aggregateHandle;

	// PT: TODO: shouldn't it be compared to mUsedSize?
	E_ASSERT(index < mVolumeData.size());

	index_ = index;
	group_ = mGroups[index];

	releaseAggregateGroup(mGroups[index]);
	resetEntry(index);

	E_ASSERT(mNbAggregates);
	mNbAggregates--;

	return true;
}

void AABBManager::handleOriginShift()
{
	mOriginShifted = false;

	// PT: TODO: isn't the following loop potentially updating removed objects?
	// PT: TODO: check that aggregates code is correct here
	for(PxU32 i=0; i<mUsedSize; i++)
	{
		if(mGroups[i] == Bp::FilterGroup::eINVALID)
			continue;

		{
			if(mVolumeData[i].isSingleActor())
			{
				if(!mAddedHandleMap.test(i))
					mUpdatedHandles.pushBack(i);	// PT: TODO: BoundsIndex-to-ShapeHandle confusion here
			}
			else if(mVolumeData[i].isAggregate())
			{
				const AggregateHandle aggregateHandle = mVolumeData[i].getAggregate();
				Aggregate* aggregate = getAggregateFromHandle(aggregateHandle);
				if(aggregate->getNbAggregated())
				{
					aggregate->markAsDirty(mDirtyAggregates);
					aggregate->allocateBounds();
					aggregate->computeBounds(mBoundsArray.begin(), mContactDistance.begin());
					mBoundsArray.begin()[aggregate->mIndex] = aggregate->getMergedBounds();
					if(!mAddedHandleMap.test(i))
						mUpdatedHandles.pushBack(i);	// PT: TODO: BoundsIndex-to-ShapeHandle confusion here
				}
			}
		}
	}
}

void AggregateBoundsComputationTask::runInternal()
{
	const BoundsArray& boundArray = mManager->getBoundsArray();
	const float* contactDistances = mManager->getContactDistances();

	PxU32 size = mNbToGo;
	Aggregate** currentAggregate = mAggregates + mStart;
	while(size--)
	{
		if(size)
		{
			Aggregate* nextAggregate = *(currentAggregate+1);
			PxPrefetchLine(nextAggregate, 0);
			PxPrefetchLine(nextAggregate, 64);
		}

		(*currentAggregate)->computeBounds(boundArray.begin(), contactDistances);
		currentAggregate++;
	}
}

void PreBpUpdateTask::runInternal()
{
	mManager->preBpUpdate_CPU(mNumCpuTasks);
}

void AABBManager::startAggregateBoundsComputationTasks(PxU32 nbToGo, PxU32 numCpuTasks, Cm::FlushPool& flushPool)
{
	const PxU32 nbAggregatesPerTask = nbToGo > numCpuTasks ? nbToGo / numCpuTasks : nbToGo;

	// PT: TODO: better load balancing

	PxU32 start = 0;
	while(nbToGo)
	{
		AggregateBoundsComputationTask* T = E_PLACEMENT_NEW(flushPool.allocate(sizeof(AggregateBoundsComputationTask)), AggregateBoundsComputationTask(mContextID));

		const PxU32 nb = nbToGo < nbAggregatesPerTask ? nbToGo : nbAggregatesPerTask;
		T->Init(this, start, nb, mDirtyAggregates.begin());
		start += nb;
		nbToGo -= nb;

		T->setContinuation(&mPreBpUpdateTask);
		T->removeReference();
	}
}

void AABBManager::updateBPFirstPass(PxU32 numCpuTasks, Cm::FlushPool& flushPool, bool /*hasContactDistanceUpdated*/, PxBaseTask* continuation)
{
	EPX_PROFILE_ZONE("AABBManager::updateBPFirstPass", mContextID);

	const bool singleThreaded = gSingleThreaded || numCpuTasks<2;
	if(!singleThreaded)
	{
		E_ASSERT(numCpuTasks);
		mPreBpUpdateTask.Init(this, numCpuTasks);
		mPreBpUpdateTask.setContinuation(continuation);
	}

	// Add
	{
		EPX_PROFILE_ZONE("AABBManager::updateBPFirstPass - add", mContextID);

		mAddedHandles.resetOrClear();

		const PxU32* bits = mAddedHandleMap.getWords();
		if(bits)
		{
			// PT: ### bitmap iterator pattern
			const PxU32 lastSetBit = mAddedHandleMap.findLast();
			for(PxU32 w = 0; w <= lastSetBit >> 5; ++w)
			{
				for(PxU32 b = bits[w]; b; b &= b-1)
				{
					const BoundsIndex handle = PxU32(w<<5|PxLowestSetBit(b));
					E_ASSERT(!mVolumeData[handle].isAggregated());
					mAddedHandles.pushBack(handle);		// PT: TODO: BoundsIndex-to-ShapeHandle confusion here
				}
			}
		}
	}

	// Update
	{
		EPX_PROFILE_ZONE("AABBManager::updateBPFirstPass - update", mContextID);

		mUpdatedHandles.resetOrClear();
		if(!mOriginShifted)
		{
			// PT: TODO:
			// - intercept calls marking aggregateD shapes dirty, in order to mark their aggregates dirty at the same time. That way we don't discover
			// them while parsing the map, i.e. the map is already fully complete when the parsing begins (no need to parse twice etc).
			// - once this is done, aggregateD shapes can be ignored during parsing (since we only needed them to go to their aggregates)
			// - we can then handle aggregates while parsing the map, i.e. there's no need for sorting anymore.
			// - there will be some thoughts to do about the dirty aggregates coming from the added map parsing: we still need to compute their bounds,
			// but we don't want to add these to mUpdatedHandles (since they're already in  mAddedHandles)
			// - we still need the set of dirty aggregates post broadphase, but we don't want to re-parse the full map for self-collisions. So we may still
			// need to put dirty aggregates in an array, but that might be simplified now
			// - the 'isDirty' checks to updatePairs can use the update map though - but the boundedTest is probably more expensive than the current test

			// PT: TODO: another idea: just output all aggregate handles by default then have a pass on mUpdatedHandles to remove them if that wasn't actually necessary
			// ...or just drop the artificial requirement for aggregates...

			{
				EPX_PROFILE_ZONE("AABBManager::updateBPFirstPass - update - bitmap iteration", mContextID);

				const PxU32* bits = mChangedHandleMap.getWords();
				if(bits)
				{
					// PT: ### bitmap iterator pattern
					const PxU32 lastSetBit = mChangedHandleMap.findLast();
					for(PxU32 w = 0; w <= lastSetBit >> 5; ++w)
					{
						for(PxU32 b = bits[w]; b; b &= b-1)
						{
							const BoundsIndex handle = PxU32(w<<5|PxLowestSetBit(b));
							E_ASSERT(!mRemovedHandleMap.test(handle));		// a handle may only be updated and deleted if it was just added.
							E_ASSERT(!mVolumeData[handle].isAggregate());	// PT: make sure changedShapes doesn't contain aggregates

							if(mAddedHandleMap.test(handle))					// just-inserted handles may also be marked updated, so skip them
								continue;

							if(mVolumeData[handle].isSingleActor())
							{
								E_ASSERT(mGroups[handle] != Bp::FilterGroup::eINVALID);
								mUpdatedHandles.pushBack(handle);	// PT: TODO: BoundsIndex-to-ShapeHandle confusion here
							}
							else
							{
								E_ASSERT(mVolumeData[handle].isAggregated());
								const AggregateHandle aggregateHandle = mVolumeData[handle].getAggregateOwner();
								Aggregate* aggregate = getAggregateFromHandle(aggregateHandle);
								// PT: an actor from the aggregate has been updated => mark dirty to recompute bounds later
								// PT: we don't recompute the bounds right away since multiple actors from the same aggregate may have changed.
								aggregate->markAsDirty(mDirtyAggregates);
							}
						}
					}
				}
			}

			const PxU32 size = mDirtyAggregates.size();
			if(size)
			{
				EPX_PROFILE_ZONE("AABBManager::updateBPFirstPass - update - dirty iteration", mContextID);
				for(PxU32 i=0;i<size;i++)
				{
					Aggregate* aggregate = mDirtyAggregates[i];
					if(i!=size-1)
					{
						Aggregate* nextAggregate = mDirtyAggregates[i];
						PxPrefetchLine(nextAggregate, 0);
					}

					aggregate->allocateBounds();
					if(singleThreaded)
					{
						aggregate->computeBounds(mBoundsArray.begin(), mContactDistance.begin());
						mBoundsArray.begin()[aggregate->mIndex] = aggregate->getMergedBounds();
					}

					// PT: Can happen when an aggregate has been created and then its actors have been changed (with e.g. setLocalPose)
					// before a BP call.
					if(!mAddedHandleMap.test(aggregate->mIndex))
						mUpdatedHandles.pushBack(aggregate->mIndex);	// PT: TODO: BoundsIndex-to-ShapeHandle confusion here
				}

				if(!singleThreaded)
					startAggregateBoundsComputationTasks(size, numCpuTasks, flushPool);

				// PT: we're already sorted if no dirty-aggregates are involved
				{
					EPX_PROFILE_ZONE("AABBManager::updateAABBsAndBP - update - sort", mContextID);

					// PT: TODO: remove this
					PxSort(mUpdatedHandles.begin(), mUpdatedHandles.size());
				}
			}
		}
		else
		{
			handleOriginShift();
		}
	}

	// Remove
	{
		EPX_PROFILE_ZONE("AABBManager::updateBPFirstPass - remove", mContextID);

		mRemovedHandles.resetOrClear();

		const PxU32* bits = mRemovedHandleMap.getWords();
		if(bits)
		{
			// PT: ### bitmap iterator pattern
			const PxU32 lastSetBit = mRemovedHandleMap.findLast();
			for(PxU32 w = 0; w <= lastSetBit >> 5; ++w)
			{
				for(PxU32 b = bits[w]; b; b &= b-1)
				{
					const BoundsIndex handle = PxU32(w<<5|PxLowestSetBit(b));
					E_ASSERT(!mVolumeData[handle].isAggregated());
					mRemovedHandles.pushBack(handle);	// PT: TODO: BoundsIndex-to-ShapeHandle confusion here
				}
			}
		}
	}

	/////

	// PT: TODO: do we need to run these threads when we origin-shifted everything before?
	if(singleThreaded)
		preBpUpdate_CPU(numCpuTasks);
	else
		mPreBpUpdateTask.removeReference();
}

// PT: previously known as AABBManager::updateAABBsAndBP
void AABBManager::updateBPSecondPass(PxcScratchAllocator* scratchAllocator, PxBaseTask* continuation)
{
	EPX_PROFILE_ZONE("AABBManager::updateBPSecondPass", mContextID);

	// PT: TODO: do we need to run these threads when we origin-shifted everything before?
	//finalizeUpdate(numCpuTasks, scratchAllocator, continuation);
	// PT: code below used to be "finalizeUpdate"

	// PT: TODO: move to base?
	const BroadPhaseUpdateData updateData(mAddedHandles.begin(), mAddedHandles.size(),
		mUpdatedHandles.begin(), mUpdatedHandles.size(),
		mRemovedHandles.begin(), mRemovedHandles.size(),
		mBoundsArray.begin(), mGroups.begin(), mContactDistance.begin(), mBoundsArray.size(),
		mFilters,
		// PT: TODO: this could also be removed now. The key to understanding the refactorings is that none of the two bools below are actualy used by the CPU versions.
		mBoundsArray.hasChanged(),
		false);

	E_ASSERT(updateData.isValid(false));

	const bool b = updateData.getNumCreatedHandles() || updateData.getNumRemovedHandles();

	//KS - skip broad phase if there are no updated shapes.
	// PT: BP UPDATE CALL
	if(b || updateData.getNumUpdatedHandles())
		mBroadPhase.update(scratchAllocator, updateData, continuation);
}

void AABBManager::preBpUpdate_CPU(PxU32 numCpuTasks)
{
	EPX_PROFILE_ZONE("AABBManager::preBpUpdate", mContextID);

	const bool singleThreaded = gSingleThreaded || numCpuTasks<2;
	if (!singleThreaded)
	{
		const PxU32 size = mDirtyAggregates.size();
		for (PxU32 i = 0; i<size; i++)
		{
			Aggregate* aggregate = mDirtyAggregates[i];
			mBoundsArray.begin()[aggregate->mIndex] = aggregate->getMergedBounds();
		}
	}
}

static EHD void outputOverlap(PxArray<AABBOverlap>* overlaps, const VolumeData* volumeData, PxU32 id0, PxU32 id1)
{
//	overlaps.pushBack(AABBOverlap(volumeData[id0].userData, volumeData[id1].userData, handle));
	const ElementType::Enum volumeType = PxMax(volumeData[id0].getVolumeType(), volumeData[id1].getVolumeType());
	//overlaps[volumeType].pushBack(AABBOverlap(reinterpret_cast<void*>(size_t(id0)), reinterpret_cast<void*>(size_t(id1))));

	AABBOverlap* overlap = Cm::reserveContainerMemory(overlaps[volumeType], 1);
	// PT: we don't convert to pointers right away because we need the IDs in postBpStage3
	overlap->mUserData0 = reinterpret_cast<void*>(size_t(id0));
	overlap->mUserData1 = reinterpret_cast<void*>(size_t(id1));
	// PT: note that overlap->mPairUserData remains uninitialized here
}

static EHD void createOverlap(PxArray<AABBOverlap>* overlaps, const VolumeData* volumeData, PxU32 id0, PxU32 id1)
{
	outputOverlap(overlaps, volumeData, id0, id1);
}

static EHD void deleteOverlap(PxArray<AABBOverlap>* overlaps, const VolumeData* volumeData, PxU32 id0, PxU32 id1)
{
//	E_ASSERT(volumeData[id0].getUserData());
//	E_ASSERT(volumeData[id1].getUserData());
	if (volumeData[id0].getUserData() && volumeData[id1].getUserData())	// PT: TODO: no idea if this is the right thing to do or if it's normal to get null ptrs here
		outputOverlap(overlaps, volumeData, id0, id1);
}

void PersistentPairs::outputDeletedOverlaps(PxArray<AABBOverlap>* overlaps, const VolumeData* volumeData)
{
	const PxU32 nbActivePairs = mPM.mNbActivePairs;
	for(PxU32 i=0;i<nbActivePairs;i++)
	{
		const InternalPair& p = mPM.mActivePairs[i];
		deleteOverlap(overlaps, volumeData, p.getId0(), p.getId1());
	}
}

EHD void PersistentPairs::updatePairs(	PxU32 timestamp, const PxBounds3* bounds, const float* contactDistances, const Bp::FilterGroup::Enum* groups,
													const bool* lut, VolumeData* volumeData, PxArray<AABBOverlap>* createdOverlaps, PxArray<AABBOverlap>* destroyedOverlaps)
{
	if(mTimestamp==timestamp)
		return;

	mTimestamp = timestamp;

	findOverlaps(mPM, bounds, contactDistances, groups, lut);

	PxU32 i=0;
	PxU32 nbActivePairs = mPM.mNbActivePairs;
	while(i<nbActivePairs)
	{
		InternalPair& p = mPM.mActivePairs[i];
		const PxU32 id0 = p.getId0();
		const PxU32 id1 = p.getId1();

		if(p.isNew())
		{
			createOverlap(createdOverlaps, volumeData, id0, id1);

			p.clearNew();
			p.clearUpdated();

			i++;
		}
		else if(p.isUpdated())
		{
			p.clearUpdated();
			i++;
		}
		else
		{
			deleteOverlap(destroyedOverlaps, volumeData, id0, id1);

			const PxU32 hashValue = hash(id0, id1) & mPM.mMask;
			mPM.removePair(id0, id1, hashValue, i);
			nbActivePairs--;
		}
	}
	mPM.shrinkMemory();
}

void AABBManager::updatePairs(PersistentPairs& p, BpCacheData* data)
{
	if (data)
		p.updatePairs(mTimestamp, mBoundsArray.begin(), mContactDistance.begin(), mGroups.begin(), mFilters.getLUT(), mVolumeData.begin(), data->mCreatedPairs, data->mDeletedPairs);
	else
		p.updatePairs(mTimestamp, mBoundsArray.begin(), mContactDistance.begin(), mGroups.begin(), mFilters.getLUT(), mVolumeData.begin(), mCreatedOverlaps, mDestroyedOverlaps);
}

static EHD Bp::FilterType::Enum convertFilterType(PxAggregateType::Enum	agType)
{
	if(agType==PxAggregateType::eGENERIC)
		return Bp::FilterType::DYNAMIC;
	else if(agType==PxAggregateType::eSTATIC)
		return Bp::FilterType::STATIC;
	E_ASSERT(agType==PxAggregateType::eKINEMATIC);
	return Bp::FilterType::KINEMATIC;
}

PersistentActorAggregatePair* AABBManager::createPersistentActorAggregatePair(ShapeHandle volA, ShapeHandle volB)
{
	ShapeHandle	actorHandle;
	ShapeHandle	aggregateHandle;
	if(mVolumeData[volA].isAggregate())
	{
		aggregateHandle = volA;
		actorHandle = volB;
	}
	else
	{
		E_ASSERT(mVolumeData[volB].isAggregate());
		aggregateHandle = volB;
		actorHandle = volA;
	}
	const AggregateHandle h = mVolumeData[aggregateHandle].getAggregate();
	Aggregate* aggregate = getAggregateFromHandle(h);
	E_ASSERT(aggregate->mIndex==aggregateHandle);

	// Single-aggregate filtering
	{
		const PxAggregateType::Enum	agType = PxGetAggregateType(aggregate->getFilterHint());
		const int t0 = convertFilterType(agType);

		const int t1 = mGroups[actorHandle] & BP_FILTERING_TYPE_MASK;	// PT: from "groupFiltering" function

		if(!mFilters.mLUT[t0][t1])
			return NULL;
	}

	return EPX_NEW(PersistentActorAggregatePair)(aggregate, actorHandle);	// PT: TODO: use a pool or something
}

PersistentAggregateAggregatePair* AABBManager::createPersistentAggregateAggregatePair(ShapeHandle volA, ShapeHandle volB)
{
	E_ASSERT(mVolumeData[volA].isAggregate() && mVolumeData[volB].isAggregate());
	const AggregateHandle h0 = mVolumeData[volA].getAggregate();
	const AggregateHandle h1 = mVolumeData[volB].getAggregate();
	Aggregate* aggregate0 = getAggregateFromHandle(h0);
	Aggregate* aggregate1 = getAggregateFromHandle(h1);
	E_ASSERT(aggregate0->mIndex==volA);
	E_ASSERT(aggregate1->mIndex==volB);

	// Aggregate-aggregate filtering
	{
		const PxAggregateType::Enum	agType0 = PxGetAggregateType(aggregate0->getFilterHint());
		const PxAggregateType::Enum	agType1 = PxGetAggregateType(aggregate1->getFilterHint());
		const Bp::FilterType::Enum t0 = convertFilterType(agType0);
		const Bp::FilterType::Enum t1 = convertFilterType(agType1);
		if(!mFilters.mLUT[t0][t1])
			return NULL;
	}

	return EPX_NEW(PersistentAggregateAggregatePair)(aggregate0, aggregate1);	// PT: TODO: use a pool or something
}

void AABBManager::processBPCreatedPair(const BroadPhasePair& pair)
{
	E_ASSERT(!mVolumeData[pair.mVolA].isAggregated());
	E_ASSERT(!mVolumeData[pair.mVolB].isAggregated());
	const bool isSingleActorA = mVolumeData[pair.mVolA].isSingleActor();
	const bool isSingleActorB = mVolumeData[pair.mVolB].isSingleActor();

	if(isSingleActorA && isSingleActorB)
	{
		createOverlap(mCreatedOverlaps, mVolumeData.begin(), pair.mVolA, pair.mVolB);	// PT: regular actor-actor pair
		return;
	}

	// PT: TODO: check if this is needed
	ShapeHandle volA = pair.mVolA;
	ShapeHandle volB = pair.mVolB;
	if(volB<volA)
		PxSwap(volA, volB);

	PersistentPairs* newPair;
	AggPairMap* pairMap;
	if(!isSingleActorA && !isSingleActorB)
	{
		pairMap = &mAggregateAggregatePairs;	// PT: aggregate-aggregate pair
		newPair = createPersistentAggregateAggregatePair(volA, volB);
	}
	else
	{
		pairMap = &mActorAggregatePairs;	// PT: actor-aggregate pair
		newPair = createPersistentActorAggregatePair(volA, volB);
	}

	if(newPair)
	{
		bool status = pairMap->insert(AggPair(volA, volB), newPair);
		E_UNUSED(status);
		E_ASSERT(status);
		updatePairs(*newPair);
	}
}

void AABBManager::processBPDeletedPair(const BroadPhasePair& pair)
{
	E_ASSERT(!mVolumeData[pair.mVolA].isAggregated());
	E_ASSERT(!mVolumeData[pair.mVolB].isAggregated());
	const bool isSingleActorA = mVolumeData[pair.mVolA].isSingleActor();
	const bool isSingleActorB = mVolumeData[pair.mVolB].isSingleActor();

	if(isSingleActorA && isSingleActorB)
	{
		deleteOverlap(mDestroyedOverlaps, mVolumeData.begin(), pair.mVolA, pair.mVolB);	// PT: regular actor-actor pair
		return;
	}

	// PT: TODO: check if this is needed
	ShapeHandle volA = pair.mVolA;
	ShapeHandle volB = pair.mVolB;
	if(volB<volA)
		PxSwap(volA, volB);

	AggPairMap* pairMap;
	if(!isSingleActorA && !isSingleActorB)
		pairMap = &mAggregateAggregatePairs;	// PT: aggregate-aggregate pair
	else
		pairMap = &mActorAggregatePairs;		// PT: actor-aggregate pair

	const AggPairMap::Entry* e = pairMap->find(AggPair(volA, volB));
	if(e && e->second)
	{
		PersistentPairs* p = e->second;
		p->outputDeletedOverlaps(mDestroyedOverlaps, mVolumeData.begin());
		p->mShouldBeDeleted = true;
	}
}

struct CreatedPairHandler
{
	static EHD void processPair(AABBManager& manager, const BroadPhasePair& pair) { manager.processBPCreatedPair(pair); }
};

struct DeletedPairHandler
{
	static EHD void processPair(AABBManager& manager, const BroadPhasePair& pair) { manager.processBPDeletedPair(pair); }
};

template<class FunctionT>
static void processBPPairs(PxU32 nbPairs, const BroadPhasePair* pairs, AABBManager& manager)
{
	// PT: TODO: figure out this ShapeHandle/BpHandle thing. Is it ok to use "BP_INVALID_BP_HANDLE" for a "ShapeHandle"?
	while(nbPairs--)
	{
		E_ASSERT(pairs->mVolA!=BP_INVALID_BP_HANDLE);
		E_ASSERT(pairs->mVolB!=BP_INVALID_BP_HANDLE);
		//if(pairs->mVolA != previousA || pairs->mVolB != previousB)
		{
			FunctionT::processPair(manager, *pairs);
		}
		pairs++;
	}
}

static void processAggregatePairs(AggPairMap& map, AABBManager& manager)
{
	// PT: TODO: hmmm we have a list of dirty aggregates but we don't have a list of dirty pairs.
	// PT: not sure how the 3.4 trunk solves this but let's just iterate all pairs for now
	// PT: atm we reuse this loop to delete removed interactions
	// PT: TODO: in fact we could handle all the "lost pairs" stuff right there with extra aabb-abb tests

	// PT: TODO: replace with decent hash map - or remove the hashmap entirely and use a linear array
	PxArray<AggPair> removedEntries;
	for(AggPairMap::Iterator iter = map.getIterator(); !iter.done(); ++iter)
	{
		PersistentPairs* p = iter->second;
		if(p->update(manager))
		{
			removedEntries.pushBack(iter->first);
			E_DELETE(p);
		}
	}
	for(PxU32 i=0;i<removedEntries.size();i++)
	{
		bool status = map.erase(removedEntries[i]);
		E_ASSERT(status);
		E_UNUSED(status);
	}
}

struct PairData
{
	PxArray<AABBOverlap>* mArray;
	PxU32 mStartIdx;
	PxU32 mCount;

	PairData() : mArray(NULL), mStartIdx(0), mCount(0)
	{
	}
};

class ProcessAggPairsBase : public Cm::Task
{
public:
	static const PxU32 MaxPairs = 16;

	PairData mCreatedPairs[2];
	PairData mDestroyedPairs[2];

	ProcessAggPairsBase(PxU64 contextID) : Cm::Task(contextID)
	{
	}

	void setCache(BpCacheData& data)
	{
		for (PxU32 i = 0; i < 2; ++i)
		{
			mCreatedPairs[i].mArray = &data.mCreatedPairs[i];
			mCreatedPairs[i].mStartIdx = data.mCreatedPairs[i].size();
			mDestroyedPairs[i].mArray = &data.mDeletedPairs[i];
			mDestroyedPairs[i].mStartIdx = data.mDeletedPairs[i].size();
		}
	}

	void updateCounters()
	{
		for (PxU32 i = 0; i < 2; ++i)
		{
			mCreatedPairs[i].mCount = mCreatedPairs[i].mArray->size() - mCreatedPairs[i].mStartIdx;
			mDestroyedPairs[i].mCount = mDestroyedPairs[i].mArray->size() - mDestroyedPairs[i].mStartIdx;
		}
	}
};

class ProcessAggPairsParallelTask : public ProcessAggPairsBase
{
public:
	PersistentPairs* mPersistentPairs[MaxPairs];
	Bp::AggPair mAggPairs[MaxPairs];
	PxU32 mNbPairs;
	AABBManager* mManager;
	AggPairMap* mMap;
	PxMutex* mMutex;
	const char* mName;

	ProcessAggPairsParallelTask(PxU64 contextID, PxMutex* mutex, AABBManager* manager, AggPairMap* map, const char* name) : ProcessAggPairsBase(contextID),
		mNbPairs(0), mManager(manager), mMap(map), mMutex(mutex), mName(name)
	{
	}

	void runInternal()
	{
		BpCacheData* data = mManager->getBpCacheData();

		setCache(*data);

		PxInlineArray<AggPair, MaxPairs> removedEntries;
		for (PxU32 i = 0; i < mNbPairs; ++i)
		{
			if(mPersistentPairs[i]->update(*mManager, data))
			{
				removedEntries.pushBack(mAggPairs[i]);
				E_DELETE(mPersistentPairs[i]);
			}
		}

		updateCounters();

		mManager->putBpCacheData(data);

		if (removedEntries.size())
		{
			PxMutex::ScopedLock lock(*mMutex);
			for (PxU32 i = 0; i < removedEntries.size(); i++)
			{
				bool status = mMap->erase(removedEntries[i]);
				E_ASSERT(status);
				E_UNUSED(status);
			}
		}
	}

	virtual const char* getName() const { return mName; }
};

class SortAggregateBoundsParallel : public Cm::Task
{
public:
	static const PxU32 MaxPairs = 16;
	Aggregate** mAggregates;
	PxU32 mNbAggs;

	SortAggregateBoundsParallel(PxU64 contextID, Aggregate** aggs, PxU32 nbAggs) : Cm::Task(contextID),
		mAggregates(aggs), mNbAggs(nbAggs)
	{
	}

	void runInternal()
	{
		EPX_PROFILE_ZONE("SortBounds", mContextID);
		for (PxU32 i = 0; i < mNbAggs; i++)
		{
			Aggregate* aggregate = mAggregates[i];

			aggregate->getSortedMinBounds();
		}
	}

	virtual const char* getName() const { return "SortAggregateBoundsParallel"; }
};

class ProcessSelfCollisionPairsParallel : public ProcessAggPairsBase
{
public:
	Aggregate** mAggregates;
	PxU32 mNbAggs;
	AABBManager* mManager;

	ProcessSelfCollisionPairsParallel(PxU64 contextID, Aggregate** aggs, PxU32 nbAggs, AABBManager* manager) : ProcessAggPairsBase(contextID),
		mAggregates(aggs), mNbAggs(nbAggs), mManager(manager)
	{
	}

	void runInternal()
	{
		BpCacheData* data = mManager->getBpCacheData();
		setCache(*data);
		EPX_PROFILE_ZONE("ProcessSelfCollisionPairs", mContextID);
		for (PxU32 i = 0; i < mNbAggs; i++)
		{
			Aggregate* aggregate = mAggregates[i];

			// PT: TODO: don't add filtered ones to this class at all!
			if(aggregate->mSelfCollisionPairs && PxGetAggregateType(aggregate->getFilterHint())!=PxAggregateType::eSTATIC)
				mManager->updatePairs(*aggregate->mSelfCollisionPairs, data);
		}
		updateCounters();
		mManager->putBpCacheData(data);
	}

	virtual const char* getName() const { return "ProcessSelfCollisionPairsParallel"; }
};

static void processAggregatePairsParallel(AggPairMap& map, AABBManager& manager, Cm::FlushPool& flushPool,
	PxBaseTask* continuation, const char* taskName, PxArray<ProcessAggPairsBase*>& pairTasks)
{
	// PT: TODO: hmmm we have a list of dirty aggregates but we don't have a list of dirty pairs.
	// PT: not sure how the 3.4 trunk solves this but let's just iterate all pairs for now
	// PT: atm we reuse this loop to delete removed interactions
	// PT: TODO: in fact we could handle all the "lost pairs" stuff right there with extra aabb-abb tests

	// PT: TODO: replace with decent hash map - or remove the hashmap entirely and use a linear array

	manager.mMapLock.lock();

	ProcessAggPairsParallelTask* task = E_PLACEMENT_NEW(flushPool.allocate(sizeof(ProcessAggPairsParallelTask)), ProcessAggPairsParallelTask)(0, &manager.mMapLock, &manager, &map, taskName);

	PxU32 startIdx = pairTasks.size();

	for (AggPairMap::Iterator iter = map.getIterator(); !iter.done(); ++iter)
	{
		task->mAggPairs[task->mNbPairs] = iter->first;
		task->mPersistentPairs[task->mNbPairs++] = iter->second;
		if (task->mNbPairs == ProcessAggPairsParallelTask::MaxPairs)
		{
			pairTasks.pushBack(task);
			task->setContinuation(continuation);
			task = E_PLACEMENT_NEW(flushPool.allocate(sizeof(ProcessAggPairsParallelTask)), ProcessAggPairsParallelTask)(0, &manager.mMapLock, &manager, &map, taskName);
		}
	}

	manager.mMapLock.unlock();

	for (PxU32 i = startIdx; i < pairTasks.size(); ++i)
	{
		pairTasks[i]->removeReference();
	}

	if (task->mNbPairs)
	{
		pairTasks.pushBack(task);
		task->setContinuation(continuation);
		task->removeReference();
		//task->runInternal();
	}
}

void AABBManager::postBroadPhase(PxBaseTask* continuation, Cm::FlushPool& flushPool)
{
	EPX_PROFILE_ZONE("AABBManager::postBroadPhase", mContextID);

	//KS - There is a continuation task for discrete broad phase, but not for CCD broad phase. PostBroadPhase for CCD broad phase runs in-line.
	//This probably should be revisited but it means that we can't use the parallel codepath within CCD.
	if (continuation)
	{
		mPostBroadPhase3.setContinuation(continuation);
		mPostBroadPhase2.setContinuation(&mPostBroadPhase3);
	}

	mTimestamp++;

	// PT: TODO: consider merging mCreatedOverlaps & mDestroyedOverlaps
	// PT: TODO: revisit memory management of mCreatedOverlaps & mDestroyedOverlaps

	// PT: this is now only used for CPU BPs so I think the fetchBroadPhaseResults call is useless here

	for(PxU32 i=0; i<ElementType::eCOUNT; i++)
	{
		mCreatedOverlaps[i].resetOrClear();
		mDestroyedOverlaps[i].resetOrClear();
	}

	{
		EPX_PROFILE_ZONE("AABBManager::postBroadPhase - process deleted pairs", mContextID);

		PxU32 nbDeletedPairs;
		const BroadPhasePair* deletedPairs = mBroadPhase.getDeletedPairs(nbDeletedPairs);
		processBPPairs<DeletedPairHandler>(nbDeletedPairs, deletedPairs, *this);
	}

	{
		//If there is a continuation task, then this is not part of CCD, so we can trigger bounds to be recomputed in parallel before pair generation runs during
		//stage 2.
		if (continuation)
		{
			const PxU32 size = mDirtyAggregates.size();
			for (PxU32 i = 0; i < size; i += SortAggregateBoundsParallel::MaxPairs)
			{
				const PxU32 nbToProcess = PxMin(size - i, SortAggregateBoundsParallel::MaxPairs);

				SortAggregateBoundsParallel* task = E_PLACEMENT_NEW(flushPool.allocate(sizeof(SortAggregateBoundsParallel)), SortAggregateBoundsParallel)
					(mContextID, &mDirtyAggregates[i], nbToProcess);

				task->setContinuation(&mPostBroadPhase2);
				task->removeReference();
			}
		}
	}

	if (continuation)
	{
		mPostBroadPhase2.setFlushPool(&flushPool);
		mPostBroadPhase3.removeReference();
		mPostBroadPhase2.removeReference();
	}
	else
	{
		postBpStage2(NULL, flushPool);
		postBpStage3(NULL);
	}
}

void AABBManager::reallocateChangedAABBMgActorHandleMap(const PxU32 size)
{
	mChangedHandleMap.resizeAndClear(size);
}

void PostBroadPhaseStage2Task::runInternal()
{
	mManager.postBpStage2(mCont, *mFlushPool);
}

void AABBManager::postBpStage2(PxBaseTask* continuation, Cm::FlushPool& flushPool)
{
	{
		const PxU32 size = mDirtyAggregates.size();
		for (PxU32 i = 0; i < size; i += ProcessSelfCollisionPairsParallel::MaxPairs)
		{
			const PxU32 nbToProcess = PxMin(size - i, ProcessSelfCollisionPairsParallel::MaxPairs);

			ProcessSelfCollisionPairsParallel* task = E_PLACEMENT_NEW(flushPool.allocate(sizeof(ProcessSelfCollisionPairsParallel)), ProcessSelfCollisionPairsParallel)
				(mContextID, &mDirtyAggregates[i], nbToProcess, this);
			if (continuation)
			{
				task->setContinuation(continuation);
				task->removeReference();
			}
			else
				task->runInternal();
			mAggPairTasks.pushBack(task);
		}
	}

	{
		if (continuation)
			processAggregatePairsParallel(mAggregateAggregatePairs, *this, flushPool, continuation, "AggAggPairs", mAggPairTasks);
		else
			processAggregatePairs(mAggregateAggregatePairs, *this);
	}

	{
		if (continuation)
			processAggregatePairsParallel(mActorAggregatePairs, *this, flushPool, continuation, "AggActorPairs", mAggPairTasks);
		else
			processAggregatePairs(mActorAggregatePairs, *this);
	}
}

void AABBManager::postBpStage3(PxBaseTask*)
{
	{
		EPX_PROFILE_ZONE("SimpleAABBManager::postBroadPhase - aggregate self-collisions", mContextID);
		const PxU32 size = mDirtyAggregates.size();
		for (PxU32 i = 0; i < size; i++)
		{
			Aggregate* aggregate = mDirtyAggregates[i];
			aggregate->resetDirtyState();

		}
		mDirtyAggregates.resetOrClear();
	}

	{
		EPX_PROFILE_ZONE("SimpleAABBManager::postBroadPhase - append pairs", mContextID);

		for (PxU32 a = 0; a < mAggPairTasks.size(); ++a)
		{
			ProcessAggPairsBase* task = mAggPairTasks[a];
			for (PxU32 t = 0; t < 2; t++)
			{
				for (PxU32 i = 0, startIdx = task->mCreatedPairs[t].mStartIdx; i < task->mCreatedPairs[t].mCount; ++i)
				{
					mCreatedOverlaps[t].pushBack((*task->mCreatedPairs[t].mArray)[i + startIdx]);
				}
				for (PxU32 i = 0, startIdx = task->mDestroyedPairs[t].mStartIdx; i < task->mDestroyedPairs[t].mCount; ++i)
				{
					mDestroyedOverlaps[t].pushBack((*task->mDestroyedPairs[t].mArray)[i + startIdx]);
				}
			}
		}

		mAggPairTasks.forceSize_Unsafe(0);

		resetBpCacheData();
	}

	{
		EPX_PROFILE_ZONE("AABBManager::postBroadPhase - process created pairs", mContextID);

		PxU32 nbCreatedPairs;
		const BroadPhasePair* createdPairs = mBroadPhase.getCreatedPairs(nbCreatedPairs);
		processBPPairs<CreatedPairHandler>(nbCreatedPairs, createdPairs, *this);
	}

	// PT: TODO: revisit this
	// Filter out pairs in mDestroyedOverlaps that already exist in mCreatedOverlaps. This should be done better using bitmaps
	// and some imposed ordering on previous operations. Later.
	// We could also have a dedicated function "reinsertBroadPhase()", which would preserve the existing interactions at Sc-level.
	if(1)
	{
		EPX_PROFILE_ZONE("AABBManager::postBroadPhase - post-process", mContextID);

		PxU32 totalCreatedOverlaps = 0;
		for (PxU32 idx=0; idx<ElementType::eCOUNT; idx++)
			totalCreatedOverlaps += mCreatedOverlaps[idx].size();

		mCreatedPairsTmp.clear();
		mCreatedPairsTmp.reserve(totalCreatedOverlaps);
		
		// PT: so this is where we convert the userData IDs to pointers
		// I don't remember why we need the mCreatedPairs hashset / this filtering pass
		// PT: TODO: why do we need to convert to ptrs at all anyway?
		for(PxU32 idx=0; idx<ElementType::eCOUNT; idx++)
		{
			{
				const PxU32 nbDestroyedOverlaps = mDestroyedOverlaps[idx].size();
				PxU32 size = mCreatedOverlaps[idx].size();
				AABBOverlap* overlaps = mCreatedOverlaps[idx].begin();
				while(size--)
				{
					const PxU32 id0 = PxU32(size_t(overlaps->mUserData0));
					const PxU32 id1 = PxU32(size_t(overlaps->mUserData1));
					overlaps->mUserData0 = mVolumeData[id0].getUserData();
					overlaps->mUserData1 = mVolumeData[id1].getUserData();
					overlaps++;
					if(nbDestroyedOverlaps)
						mCreatedPairsTmp.insert(Pair(id0, id1));
				}
			}
			{
				AABBOverlap* overlapsSrc = mDestroyedOverlaps[idx].begin();
				AABBOverlap* overlapsDst = overlapsSrc;

				PxU32 size = mDestroyedOverlaps[idx].size();
				PxU32 newSize = 0;
				while(size--)
				{
					const PxU32 id0 = PxU32(size_t(overlapsSrc->mUserData0));
					const PxU32 id1 = PxU32(size_t(overlapsSrc->mUserData1));
					overlapsSrc++;

					if(!mCreatedPairsTmp.contains(Pair(id0, id1)))
					{
						overlapsDst->mUserData0 = mVolumeData[id0].getUserData();
						overlapsDst->mUserData1 = mVolumeData[id1].getUserData();
						overlapsDst++;
						newSize++;
					}
				}
				mDestroyedOverlaps[idx].forceSize_Unsafe(newSize);
			}
		}
	}

	// Handle out-of-bounds objects
	{
		EPX_PROFILE_ZONE("AABBManager::postBroadPhase - out-of-bounds", mContextID);
		PxU32 nbObjects = mBroadPhase.getNbOutOfBoundsObjects();
		const PxU32* objects = mBroadPhase.getOutOfBoundsObjects();
		while(nbObjects--)
		{
			const PxU32 index = *objects++;
			if(!mRemovedHandleMap.test(index))
			{
				if(mVolumeData[index].isSingleActor())
					mOutOfBoundsObjects.pushBack(mVolumeData[index].getUserData());
				else
				{
					E_ASSERT(mVolumeData[index].isAggregate());
					mOutOfBoundsAggregates.pushBack(mVolumeData[index].getUserData());
				}
			}
		}
	}

	{
		EPX_PROFILE_ZONE("AABBManager::postBroadPhase - clear", mContextID);
		mAddedHandleMap.clear();
		mRemovedHandleMap.clear();
	}
}

BpCacheData* AABBManager::getBpCacheData()
{
	BpCacheData* rv = static_cast<BpCacheData*>(mBpThreadContextPool.pop());
	if (rv == NULL)
	{
		rv = E_PLACEMENT_NEW(EPX_ALLOC(sizeof(BpCacheData), "BpCacheData"), BpCacheData)();
	}
	return rv;
}

void AABBManager::putBpCacheData(BpCacheData* data)
{
	mBpThreadContextPool.push(*data);
}

void AABBManager::resetBpCacheData()
{
	PxInlineArray<BpCacheData*, 16> bpCache;
	BpCacheData* entry = static_cast<BpCacheData*>(mBpThreadContextPool.pop());
	while (entry)
	{
		entry->reset();
		bpCache.pushBack(entry);
		entry = static_cast<BpCacheData*>(mBpThreadContextPool.pop());
	}

	//Now reinsert back into queue...
	for (PxU32 i = 0; i < bpCache.size(); ++i)
	{
		mBpThreadContextPool.push(*bpCache[i]);
	}
}

bool AABBManager::getOutOfBoundsObjects(OutOfBoundsData& data)
{
	data.mNbOutOfBoundsObjects = mOutOfBoundsObjects.size();
	data.mOutOfBoundsObjects = mOutOfBoundsObjects.begin();
	data.mNbOutOfBoundsAggregates = mOutOfBoundsAggregates.size();
	data.mOutOfBoundsAggregates = mOutOfBoundsAggregates.begin();
	return data.mNbOutOfBoundsObjects || data.mNbOutOfBoundsAggregates;
}

void AABBManager::clearOutOfBoundsObjects()
{
	mOutOfBoundsObjects.clear();
	mOutOfBoundsAggregates.clear();
}

// PT: disabled this by default, since it bypasses all per-shape/per-actor visualization flags
//static const bool gVisualizeAggregateElems = false;
void AABBManager::visualize(PxRenderOutput& out)
{
	const PxTransform idt = PxTransform(PxIdentity);
	out << idt;

	PxBitMap bitmap;
	buildFreeBitmap(bitmap, mFirstFreeAggregate, mAggregates);

	const PxU32 N = mAggregates.size();
	for(PxU32 i=0;i<N;i++)
	{
		if(bitmap.test(i))
			continue;

		Aggregate* aggregate = mAggregates[i];
		if(aggregate->getNbAggregated())
		{
			out << PxU32(PxDebugColor::eARGB_GREEN);
			const PxBounds3& b = mBoundsArray.getBounds(aggregate->mIndex);
			renderOutputDebugBox(out, b);
		}
	}

/*	const PxU32 N = mAggregateManager.getAggregatesCapacity();
	for(PxU32 i=0;i<N;i++)
	{
		const Aggregate* aggregate = mAggregateManager.getAggregate(i);
		if(aggregate->nbElems)
		{
			if(!mAggregatesUpdated.isInList(BpHandle(i)))
				out << PxU32(PxDebugColor::eARGB_GREEN);
			else
				out << PxU32(PxDebugColor::eARGB_RED);

			PxBounds3 decoded;
			const IntegerAABB& iaabb = mBPElems.getAABB(aggregate->bpElemId);
			iaabb.decode(decoded);

			out << DebugBox(decoded, true);

			if(gVisualizeAggregateElems)
			{
				PxU32 elem = aggregate->elemHeadID;
				while(BP_INVALID_BP_HANDLE!=elem)
				{
					out << PxU32(PxDebugColor::eARGB_CYAN);
					const IntegerAABB elemBounds = mAggregateElems.getAABB(elem);
					elemBounds.decode(decoded);
					out << DebugBox(decoded, true);
					elem = mAggregateElems.getNextId(elem);
				}
			}
		}
	}*/
}

} //namespace Bp

} //namespace px

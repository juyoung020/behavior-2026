// ===== 원본: physx/include/foundation/PxBitUtils.h
namespace px
{
EHDI uint32_t PxBitCount(uint32_t v)
{
	// from http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
	uint32_t const w = v - ((v >> 1) & 0x55555555);
	uint32_t const x = (w & 0x33333333) + ((w >> 2) & 0x33333333);
	return (((x + (x >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}

EHDI bool PxIsPowerOfTwo(uint32_t x)
{
	return x != 0 && (x & (x - 1)) == 0;
}

// "Next Largest Power of 2
// Given a binary integer value x, the next largest power of 2 can be computed by a SWAR algorithm
// that recursively "folds" the upper bits into the lower bits. This process yields a bit vector with
// the same most significant 1 as x, but all 1's below it. Adding 1 to that value yields the next
// largest power of 2. For a 32-bit value:"
EHDI uint32_t PxNextPowerOfTwo(uint32_t x)
{
	x |= (x >> 1);
	x |= (x >> 2);
	x |= (x >> 4);
	x |= (x >> 8);
	x |= (x >> 16);
	return x + 1;
}

/*!
Return the index of the highest set bit. Not valid for zero arg.
*/

EHDI uint32_t PxLowestSetBit(uint32_t x)
{
	E_ASSERT(x);
	return PxLowestSetBitUnsafe(x);
}

/*!
Return the index of the highest set bit. Not valid for zero arg.
*/

EHDI uint32_t PxLowestSetBit(uint64_t x)
{
	E_ASSERT(x);
	return PxLowestSetBitUnsafe(x);
}

/*!
Return the index of the highest set bit. Not valid for zero arg.
*/

EHDI uint32_t PxHighestSetBit(uint32_t x)
{
	E_ASSERT(x);
	return PxHighestSetBitUnsafe(x);
}

/*!
Return the index of the highest set bit. Not valid for zero arg.
*/
EHDI uint32_t PxHighestSetBit(uint64_t x)
{
	E_ASSERT(x);
	return PxHighestSetBitUnsafe(x);
}

// Helper function to approximate log2 of an integer value
// assumes that the input is actually power of two.
EHDI uint32_t PxILog2(uint32_t num)
{
	for(uint32_t i = 0; i < 32; i++)
	{
		num >>= 1;
		if(num == 0)
			return i;
	}

	E_ASSERT(0);
	return uint32_t(-1);
}

} // namespace px

// ===== 원본: physx/include/foundation/PxHash.h
/*!
Central definition of hash functions
*/

namespace px
{
// Hash functions

// Thomas Wang's 32 bit mix
// http://www.cris.com/~Ttwang/tech/inthash.htm
EHD uint32_t PxComputeHash(const uint32_t key)
{
	uint32_t k = key;
	k += ~(k << 15);
	k ^= (k >> 10);
	k += (k << 3);
	k ^= (k >> 6);
	k += ~(k << 11);
	k ^= (k >> 16);
	return uint32_t(k);
}

EHD uint32_t PxComputeHash(const int32_t key)
{
	return PxComputeHash(uint32_t(key));
}

// Thomas Wang's 64 bit mix
// http://www.cris.com/~Ttwang/tech/inthash.htm
EHD uint32_t PxComputeHash(const uint64_t key)
{
	uint64_t k = key;
	k += ~(k << 32);
	k ^= (k >> 22);
	k += ~(k << 13);
	k ^= (k >> 8);
	k += (k << 3);
	k ^= (k >> 15);
	k += ~(k << 27);
	k ^= (k >> 31);
	return uint32_t(UINT32_MAX & k);
}


// Hash function for pointers
EHDI uint32_t PxComputeHash(const void* ptr)
{
	return PxComputeHash(uint64_t(ptr));
}

// Hash function for pairs
template <typename F, typename S>
EHDI uint32_t PxComputeHash(const PxPair<F, S>& p)
{
	uint32_t seed = 0x876543;
	uint32_t m = 1000007;
	return PxComputeHash(p.second) ^ (m * (PxComputeHash(p.first) ^ (m * seed)));
}

// hash object for hash map template parameter
template <class Key>
struct PxHash
{
	EHDI uint32_t operator()(const Key& k) const
	{
		return PxComputeHash(k);
	}
	EHDI bool equal(const Key& k0, const Key& k1) const
	{
		return k0 == k1;
	}
};

// specialization for strings
template <>
struct PxHash<const char*>
{
  public:
	EHDI uint32_t operator()(const char* _string) const
	{
		// "DJB" string hash
		const uint8_t* string = reinterpret_cast<const uint8_t*>(_string);
		uint32_t h = 5381;
		for(const uint8_t* ptr = string; *ptr; ptr++)
			h = ((h << 5) + h) ^ uint32_t(*ptr);
		return h;
	}
	EHDI bool equal(const char* string0, const char* string1) const
	{
		return !Pxstrcmp(string0, string1);
	}
};

} // namespace px

// ===== 원본: physx/source/common/src/CmRadixSort.h
namespace px
{
namespace Cm
{
	enum RadixHint
	{
		RADIX_SIGNED,		//!< Input values are signed
		RADIX_UNSIGNED,		//!< Input values are unsigned

		RADIX_FORCE_DWORD = 0x7fffffff
	};

#define INVALIDATE_RANKS	mCurrentSize|=0x80000000
#define VALIDATE_RANKS		mCurrentSize&=0x7fffffff
#define CURRENT_SIZE		(mCurrentSize&0x7fffffff)
#define INVALID_RANKS		(mCurrentSize&0x80000000)

	class  RadixSort
	{
										E_NOCOPY(RadixSort)
		public:
										EHDV RadixSort();
		EHDV virtual							~RadixSort();
		// Sorting methods
						EHDV RadixSort&		Sort(const PxU32* input, PxU32 nb, RadixHint hint=RADIX_SIGNED);
						EHDV RadixSort&		Sort(const float* input, PxU32 nb);

		//! Access to results. mRanks is a list of indices in sorted order, i.e. in the order you may further process your data
		EHD	const PxU32*	GetRanks()			const	{ return mRanks;		}

		//! mIndices2 gets trashed on calling the sort routine, but otherwise you can recycle it the way you want.
		EHD	PxU32*			GetRecyclable()		const	{ return mRanks2;		}

		//! Returns the total number of calls to the radix sorter.
		EHD	PxU32			GetNbTotalCalls()	const	{ return mTotalCalls;	}
		//! Returns the number of eraly exits due to temporal coherence.
		EHD	PxU32			GetNbHits()			const	{ return mNbHits;		}

		EHD	void			invalidateRanks()			{ INVALIDATE_RANKS;		}

						EHDV bool			SetBuffers(PxU32* ranks0, PxU32* ranks1, PxU32* histogram1024, PxU32** links256);
		protected:
						PxU32			mCurrentSize;		//!< Current size of the indices list
						PxU32*			mRanks;				//!< Two lists, swapped each pass
						PxU32*			mRanks2;
						PxU32*			mHistogram1024;
						PxU32**			mLinks256;
		// Stats
						PxU32			mTotalCalls;		//!< Total number of calls to the sort routine
						PxU32			mNbHits;			//!< Number of early exits due to coherence

						// Stack-radix
						bool			mDeleteRanks;		//!<
	};

	#define StackRadixSort(name, ranks0, ranks1)	\
		RadixSort name;								\
		PxU32 histogramBuffer[1024];				\
		PxU32* linksBuffer[256];					\
		name.SetBuffers(ranks0, ranks1, histogramBuffer, linksBuffer);

	class  RadixSortBuffered : public RadixSort
	{
	public:
							EHDV RadixSortBuffered();
							EHDV ~RadixSortBuffered();

		EHDV void				reset();

		EHDV RadixSortBuffered&	Sort(const PxU32* input, PxU32 nb, RadixHint hint=RADIX_SIGNED);
		EHDV RadixSortBuffered&	Sort(const float* input, PxU32 nb);

	private:
							EHDV RadixSortBuffered(const RadixSortBuffered& object);
							RadixSortBuffered& operator=(const RadixSortBuffered& object);

		// Internal methods
		EHDV void				CheckResize(PxU32 nb);
		EHDV bool				Resize(PxU32 nb);
	};
}
}

// ===== 원본: physx/source/common/src/CmRadixSort.cpp
// PT: code archeology: this initially came from ICE (IceRevisitedRadix.h/cpp). Consider putting it back the way it was initially.

using namespace px;
using namespace Cm;

	#define H0_OFFSET	0
	#define H1_OFFSET	256
	#define H2_OFFSET	512
	#define H3_OFFSET	768
	#define BYTES_INC	j

#define CREATE_HISTOGRAMS(type, buffer)														\
	/* Clear counters/histograms */															\
	PxMemZero(mHistogram1024, 256*4*sizeof(PxU32));											\
																							\
	/* Prepare to count */																	\
	const PxU8*  p = reinterpret_cast<const PxU8*>(input);						\
	const PxU8*  pe = &p[nb*4];													\
	PxU32*  h0= &mHistogram1024[H0_OFFSET];	/* Histogram for first pass (LSB)*/	\
	PxU32*  h1= &mHistogram1024[H1_OFFSET];	/* Histogram for second pass	*/	\
	PxU32*  h2= &mHistogram1024[H2_OFFSET];	/* Histogram for third pass		*/	\
	PxU32*  h3= &mHistogram1024[H3_OFFSET];	/* Histogram for last pass (MSB)*/	\
																							\
	bool AlreadySorted = true;	/* Optimism... */											\
																							\
	if(INVALID_RANKS)																		\
	{																						\
		/* Prepare for temporal coherence */												\
		const type*  Running = reinterpret_cast<const type*>(buffer);			\
		type PrevVal = *Running;															\
																							\
		while(p!=pe)																		\
		{																					\
			/* Read input buffer in previous sorted order */								\
			const type Val = *Running++;													\
			/* Check whether already sorted or not */										\
			if(Val<PrevVal)	{ AlreadySorted = false; break; } /* Early out */				\
			/* Update for next iteration */													\
			PrevVal = Val;																	\
																							\
			/* Create histograms */															\
			h0[*p++]++;	h1[*p++]++;	h2[*p++]++;	h3[*p++]++;									\
		}																					\
																							\
		/* If all input values are already sorted, we just have to return and leave the */	\
		/* previous list unchanged. That way the routine may take advantage of temporal */	\
		/* coherence, for example when used to sort transparent faces.					*/	\
		if(AlreadySorted)																	\
		{																					\
			mNbHits++;																		\
			for(PxU32 i=0;i<nb;i++)	mRanks[i] = i;											\
			return *this;																	\
		}																					\
	}																						\
	else																					\
	{																						\
		/* Prepare for temporal coherence */												\
		const PxU32*  Indices = mRanks;											\
		type PrevVal = type(buffer[*Indices]);												\
																							\
		while(p!=pe)																		\
		{																					\
			/* Read input buffer in previous sorted order */								\
			const type Val = type(buffer[*Indices++]);										\
			/* Check whether already sorted or not */										\
			if(Val<PrevVal)	{ AlreadySorted = false; break; } /* Early out */				\
			/* Update for next iteration */													\
			PrevVal = Val;																	\
																							\
			/* Create histograms */															\
			h0[*p++]++;	h1[*p++]++;	h2[*p++]++;	h3[*p++]++;									\
		}																					\
																							\
		/* If all input values are already sorted, we just have to return and leave the */	\
		/* previous list unchanged. That way the routine may take advantage of temporal */	\
		/* coherence, for example when used to sort transparent faces.					*/	\
		if(AlreadySorted)	{ mNbHits++; return *this;	}									\
	}																						\
																							\
	/* Else there has been an early out and we must finish computing the histograms */		\
	while(p!=pe)																			\
	{																						\
		/* Create histograms without the previous overhead */								\
		h0[*p++]++;	h1[*p++]++;	h2[*p++]++;	h3[*p++]++;										\
	}

EHDI const PxU32* CheckPassValidity(PxU32 pass, const PxU32* mHistogram1024, PxU32 nb, const void* input, PxU8& UniqueVal)
{
	// Shortcut to current counters
	const PxU32* CurCount = &mHistogram1024[pass<<8];

	// Check pass validity

	// If all values have the same byte, sorting is useless.
	// It may happen when sorting bytes or words instead of dwords.
	// This routine actually sorts words faster than dwords, and bytes
	// faster than words. Standard running time (O(4*n))is reduced to O(2*n)
	// for words and O(n) for bytes. Running time for floats depends on actual values...

	// Get first byte
	UniqueVal = *((reinterpret_cast<const PxU8*>(input))+pass);

	// Check that byte's counter
	if(CurCount[UniqueVal]==nb)
		return NULL;

	return CurCount;
}

EHDI RadixSort::RadixSort() : mCurrentSize(0), mRanks(NULL), mRanks2(NULL), mHistogram1024(0), mLinks256(0), mTotalCalls(0), mNbHits(0), mDeleteRanks(true)
{
	// Initialize indices
	INVALIDATE_RANKS;
}

EHDI RadixSort::~RadixSort()
{
}

/**
 *	Main sort routine.
 *	This one is for integer values. After the call, mRanks contains a list of indices in sorted order, i.e. in the order you may process your data.
 *	\param		input	[in] a list of integer values to sort
 *	\param		nb		[in] number of values to sort, must be < 2^31
 *	\param		hint	[in] RADIX_SIGNED to handle negative values, RADIX_UNSIGNED if you know your input buffer only contains positive values
 *	\return		Self-Reference
 */
EHDI RadixSort& RadixSort::Sort(const PxU32* input, PxU32 nb, RadixHint hint)
{
	E_ASSERT(mHistogram1024);
	E_ASSERT(mLinks256);
	E_ASSERT(mRanks);
	E_ASSERT(mRanks2);

	// Checkings
	if(!input || !nb || nb&0x80000000)
		return *this;

	// Stats
	mTotalCalls++;

	// Create histograms (counters). Counters for all passes are created in one run.
	// Pros:	read input buffer once instead of four times
	// Cons:	mHistogram1024 is 4Kb instead of 1Kb
	// We must take care of signed/unsigned values for temporal coherence.... I just
	// have 2 code paths even if just a single opcode changes. Self-modifying code, someone?
	if(hint==RADIX_UNSIGNED)	{ CREATE_HISTOGRAMS(PxU32, input);	}
	else						{ CREATE_HISTOGRAMS(PxI32, input);	}

	// Compute #negative values involved if needed
	PxU32 NbNegativeValues = 0;
	if(hint==RADIX_SIGNED)
	{
		// An efficient way to compute the number of negatives values we'll have to deal with is simply to sum the 128
		// last values of the last histogram. Last histogram because that's the one for the Most Significant Byte,
		// responsible for the sign. 128 last values because the 128 first ones are related to positive numbers.
		PxU32*  h3= &mHistogram1024[768];
		for(PxU32 i=128;i<256;i++)	NbNegativeValues += h3[i];	// 768 for last histogram, 128 for negative part
	}

	// Radix sort, j is the pass number (0=LSB, 3=MSB)
	for(PxU32 j=0;j<4;j++)
	{
//		CHECK_PASS_VALIDITY(j);
		PxU8 UniqueVal;
		const PxU32*  CurCount = CheckPassValidity(j, mHistogram1024, nb, input, UniqueVal);

		// Sometimes the fourth (negative) pass is skipped because all numbers are negative and the MSB is 0xFF (for example). This is
		// not a problem, numbers are correctly sorted anyway.
		if(CurCount)
		{
			PxU32**  Links256 = mLinks256;

			// Should we care about negative values?
			if(j!=3 || hint==RADIX_UNSIGNED)
			{
				// Here we deal with positive values only

				// Create offsets
				Links256[0] = mRanks2;
				for(PxU32 i=1;i<256;i++)
					Links256[i] = Links256[i-1] + CurCount[i-1];
			}
			else
			{
				// This is a special case to correctly handle negative integers. They're sorted in the right order but at the wrong place.

				// Create biased offsets, in order for negative numbers to be sorted as well
				Links256[0] = &mRanks2[NbNegativeValues];										// First positive number takes place after the negative ones
				for(PxU32 i=1;i<128;i++)
					Links256[i] = Links256[i-1] + CurCount[i-1];		// 1 to 128 for positive numbers

				// Fixing the wrong place for negative values
				Links256[128] = mRanks2;
				for(PxU32 i=129;i<256;i++)
					Links256[i] = Links256[i-1] + CurCount[i-1];
			}

			// Perform Radix Sort
			const PxU8*  InputBytes = reinterpret_cast<const PxU8*>(input);
            InputBytes += BYTES_INC;
			if(INVALID_RANKS)
			{
				for(PxU32 i=0;i<nb;i++)
					*Links256[InputBytes[i<<2]]++ = i;
				VALIDATE_RANKS;
			}
			else
			{
				PxU32*  Indices		= mRanks;
				PxU32*  IndicesEnd	= &mRanks[nb];
				while(Indices!=IndicesEnd)
				{
					const PxU32 id = *Indices++;
					*Links256[InputBytes[id<<2]]++ = id;
				}
			}

			// Swap pointers for next pass. Valid indices - the most recent ones - are in mRanks after the swap.
			PxU32* Tmp	= mRanks;	mRanks = mRanks2; mRanks2 = Tmp;
		}
	}
	return *this;
}

/**
 *	Main sort routine.
 *	This one is for floating-point values. After the call, mRanks contains a list of indices in sorted order, i.e. in the order you may process your data.
 *	\param		input2			[in] a list of floating-point values to sort
 *	\param		nb				[in] number of values to sort, must be < 2^31
 *	\return		Self-Reference
 *	\warning	only sorts IEEE floating-point values
 */
EHDI RadixSort& RadixSort::Sort(const float* input2, PxU32 nb)
{
	E_ASSERT(mHistogram1024);
	E_ASSERT(mLinks256);
	E_ASSERT(mRanks);
	E_ASSERT(mRanks2);

	// Checkings
	if(!input2 || !nb || nb&0x80000000)
		return *this;

	// Stats
	mTotalCalls++;

	const PxU32*  input = reinterpret_cast<const PxU32*>(input2);

	// Allocate histograms & offsets on the stack
	//PxU32 mHistogram1024[256*4];
	//PxU32* mLinks256[256];

	// Create histograms (counters). Counters for all passes are created in one run.
	// Pros:	read input buffer once instead of four times
	// Cons:	mHistogram1024 is 4Kb instead of 1Kb
	// Floating-point values are always supposed to be signed values, so there's only one code path there.
	// Please note the floating point comparison needed for temporal coherence! Although the resulting asm code
	// is dreadful, this is surprisingly not such a performance hit - well, I suppose that's a big one on first
	// generation Pentiums....We can't make comparison on integer representations because, as Chris said, it just
	// wouldn't work with mixed positive/negative values....
	{ CREATE_HISTOGRAMS(float, input2); }

	// Compute #negative values involved if needed
	PxU32 NbNegativeValues = 0;
	// An efficient way to compute the number of negatives values we'll have to deal with is simply to sum the 128
	// last values of the last histogram. Last histogram because that's the one for the Most Significant Byte,
	// responsible for the sign. 128 last values because the 128 first ones are related to positive numbers.
	// ### is that ok on Apple ?!
	PxU32*  h3= &mHistogram1024[768];
	for(PxU32 i=128;i<256;i++)	NbNegativeValues += h3[i];	// 768 for last histogram, 128 for negative part

	// Radix sort, j is the pass number (0=LSB, 3=MSB)
	for(PxU32 j=0;j<4;j++)
	{
		PxU8 UniqueVal;
		const PxU32*  CurCount = CheckPassValidity(j, mHistogram1024, nb, input, UniqueVal);
		// Should we care about negative values?
		if(j!=3)
		{
			// Here we deal with positive values only
//			CHECK_PASS_VALIDITY(j);
//			const bool PerformPass = CheckPassValidity(j, mHistogram1024, nb, input);

			if(CurCount)
			{
				PxU32**  Links256 = mLinks256;

				// Create offsets
				Links256[0] = mRanks2;
				for(PxU32 i=1;i<256;i++)
					Links256[i] = Links256[i-1] + CurCount[i-1];

				// Perform Radix Sort
				const PxU8*  InputBytes = reinterpret_cast<const PxU8*>(input);
                InputBytes += BYTES_INC;
				if(INVALID_RANKS)
				{
					for(PxU32 i=0;i<nb;i++)
						*Links256[InputBytes[i<<2]]++ = i;
					VALIDATE_RANKS;
				}
				else
				{
					PxU32*  Indices		= mRanks;
					PxU32*  IndicesEnd	= &mRanks[nb];
					while(Indices!=IndicesEnd)
					{
						const PxU32 id = *Indices++;
						*Links256[InputBytes[id<<2]]++ = id;
					}
				}

				// Swap pointers for next pass. Valid indices - the most recent ones - are in mRanks after the swap.
				PxU32* Tmp	= mRanks;	mRanks = mRanks2; mRanks2 = Tmp;
			}
		}
		else
		{
			// This is a special case to correctly handle negative values
//			CHECK_PASS_VALIDITY(j);
//			const bool PerformPass = CheckPassValidity(j, mHistogram1024, nb, input);

			if(CurCount)
			{
				PxU32**  Links256 = mLinks256;

				// Create biased offsets, in order for negative numbers to be sorted as well
				Links256[0] = &mRanks2[NbNegativeValues];										// First positive number takes place after the negative ones
				for(PxU32 i=1;i<128;i++)
					Links256[i] = Links256[i-1] + CurCount[i-1];		// 1 to 128 for positive numbers

				// We must reverse the sorting order for negative numbers!
				Links256[255] = mRanks2;
				for(PxU32 i=0;i<127;i++)
					Links256[254-i] = Links256[255-i] + CurCount[255-i];		// Fixing the wrong order for negative values
				for(PxU32 i=128;i<256;i++)
					Links256[i] += CurCount[i];							// Fixing the wrong place for negative values

				// Perform Radix Sort
				if(INVALID_RANKS)
				{
					for(PxU32 i=0;i<nb;i++)
					{
						const PxU32 Radix = input[i]>>24;				// Radix byte, same as above. AND is useless here (PxU32).
						// ### cmp to be killed. Not good. Later.
						if(Radix<128)	*Links256[Radix]++ = i;			// Number is positive, same as above
						else			*(--Links256[Radix]) = i;		// Number is negative, flip the sorting order
					}
					VALIDATE_RANKS;
				}
				else
				{
					const PxU32*  Ranks = mRanks;
					for(PxU32 i=0;i<nb;i++)
					{
						const PxU32 Radix = input[Ranks[i]]>>24;				// Radix byte, same as above. AND is useless here (PxU32).
                        // ### cmp to be killed. Not good. Later.
						if(Radix<128)	*Links256[Radix]++ = Ranks[i];			// Number is positive, same as above
						else			*(--Links256[Radix]) = Ranks[i];		// Number is negative, flip the sorting order
					}
				}
				// Swap pointers for next pass. Valid indices - the most recent ones - are in mRanks after the swap.
				PxU32* Tmp	= mRanks;	mRanks = mRanks2; mRanks2 = Tmp;
			}
			else
			{
				// The pass is useless, yet we still have to reverse the order of current list if all values are negative.
				if(UniqueVal>=128)
				{
					if(INVALID_RANKS)
					{
						// ###Possible?
						for(PxU32 i=0;i<nb;i++)	mRanks2[i] = nb-i-1;
						VALIDATE_RANKS;
					}
					else
					{
						for(PxU32 i=0;i<nb;i++)	mRanks2[i] = mRanks[nb-i-1];
					}

					// Swap pointers for next pass. Valid indices - the most recent ones - are in mRanks after the swap.
					PxU32* Tmp	= mRanks;	mRanks = mRanks2; mRanks2 = Tmp;
				}
			}
		}
	}
	return *this;
}

EHDI bool RadixSort::SetBuffers(PxU32* ranks0, PxU32* ranks1, PxU32* histogram1024, PxU32** links256)
{
	if(!ranks0 || !ranks1 || !histogram1024 || !links256)
		return false;

	mRanks			= ranks0;
	mRanks2			= ranks1;
	mHistogram1024	= histogram1024;
	mLinks256		= links256;
	mDeleteRanks	= false;
	INVALIDATE_RANKS;
	return true;
}



using namespace px;
using namespace Cm;

EHDI RadixSortBuffered::RadixSortBuffered()
: RadixSort()
{
}

EHDI RadixSortBuffered::~RadixSortBuffered()
{
	reset();
}

EHDI void RadixSortBuffered::reset()
{
	// Release everything
	if(mDeleteRanks)
	{
		E_FREE(mRanks2);
		E_FREE(mRanks);
	}
	mCurrentSize = 0;
	INVALIDATE_RANKS;
}

/**
 *	Resizes the inner lists.
 *	\param		nb	[in] new size (number of dwords)
 *	\return		true if success
 */
EHDI bool RadixSortBuffered::Resize(PxU32 nb)
{
	if(mDeleteRanks)
	{
		// Free previously used ram
		E_FREE(mRanks2);
		E_FREE(mRanks);

		// Get some fresh one
		mRanks	= EPX_ALLOCATE(PxU32, nb, "RadixSortBuffered:mRanks");
		mRanks2	= EPX_ALLOCATE(PxU32, nb, "RadixSortBuffered:mRanks2");
	}

	return true;
}

EHDI void RadixSortBuffered::CheckResize(PxU32 nb)
{
	PxU32 CurSize = CURRENT_SIZE;
	if(nb!=CurSize)
	{
		if(nb>CurSize)
			Resize(nb);
		mCurrentSize = nb;
		INVALIDATE_RANKS;
	}
}

/**
 *	Main sort routine.
 *	This one is for integer values. After the call, mRanks contains a list of indices in sorted order, i.e. in the order you may process your data.
 *	\param		input	[in] a list of integer values to sort
 *	\param		nb		[in] number of values to sort, must be < 2^31
 *	\param		hint	[in] RADIX_SIGNED to handle negative values, RADIX_UNSIGNED if you know your input buffer only contains positive values
 *	\return		Self-Reference
 */
EHDI RadixSortBuffered& RadixSortBuffered::Sort(const PxU32* input, PxU32 nb, RadixHint hint)
{
	// Checkings
	if(!input || !nb || nb&0x80000000)
		return *this;

	// Resize lists if needed
	CheckResize(nb);

	//Set histogram buffers.
	PxU32 histogram[1024];
	PxU32* links[256];
	mHistogram1024 = histogram;
	mLinks256 = links;

	RadixSort::Sort(input, nb, hint);
	return *this;
}

/**
 *	Main sort routine.
 *	This one is for floating-point values. After the call, mRanks contains a list of indices in sorted order, i.e. in the order you may process your data.
 *	\param		input2			[in] a list of floating-point values to sort
 *	\param		nb				[in] number of values to sort, must be < 2^31
 *	\return		Self-Reference
 *	\warning	only sorts IEEE floating-point values
 */
EHDI RadixSortBuffered& RadixSortBuffered::Sort(const float* input2, PxU32 nb)
{
	// Checkings
	if(!input2 || !nb || nb&0x80000000)
		return *this;

	// Resize lists if needed
	CheckResize(nb);

	//Set histogram buffers.
	PxU32 histogram[1024];
	PxU32* links[256];
	mHistogram1024 = histogram;
	mLinks256 = links;

	RadixSort::Sort(input2, nb);
	return *this;
}

// ===== 원본: physx/source/lowlevelaabb/include/BpFiltering.h
namespace px
{
	namespace Bp
	{

#define BP_USE_AGGREGATE_GROUP_TAIL	1
#define BP_FILTERING_TYPE_SHIFT_BIT	3
#define BP_FILTERING_TYPE_MASK		7

	/*
	\brief AABBManager volumes with the same filter group value are guaranteed never to generate an overlap pair.
	\note To ensure that static pairs never overlap, add static shapes with eSTATICS.
	The value eDYNAMICS_BASE provides a minimum recommended group value for dynamic shapes.
	If dynamics shapes are assigned group values greater than or equal to eDYNAMICS_BASE then
	they are allowed to generate broadphase overlaps with statics, and other dynamic shapes provided 
	they have different group values.
	\see AABBManager::createVolume
	*/
	struct FilterGroup
	{
		enum Enum
		{
			eSTATICS		= 0,
			eDYNAMICS_BASE	= 1,
			eAGGREGATE_BASE	= 0xfffffffe,
			eINVALID		= 0xffffffff
		};
	};

	struct FilterType
	{
		enum Enum
		{
			STATIC				= 0,
			KINEMATIC			= 1,
			DYNAMIC				= 2,
			AGGREGATE			= 3,
			DEFORMABLE_SURFACE	= 4,
			DEFORMABLE_VOLUME	= 5,
			PARTICLESYSTEM		= 6,

			COUNT				= 7
		};
	};

	EHD Bp::FilterGroup::Enum getFilterGroup_Statics()
	{
		return Bp::FilterGroup::eSTATICS;
	}

	EHD	Bp::FilterGroup::Enum	getFilterGroup_Dynamics(PxU32 rigidId, bool isKinematic)
	{
		const PxU32 group = rigidId + Bp::FilterGroup::eDYNAMICS_BASE;
		const PxU32 type = isKinematic ? FilterType::KINEMATIC : FilterType::DYNAMIC;
		return Bp::FilterGroup::Enum((group<< BP_FILTERING_TYPE_SHIFT_BIT)|type);
	}

	EHD	Bp::FilterGroup::Enum	getFilterGroup(bool isStatic, PxU32 rigidId, bool isKinematic)
	{
		return isStatic ? getFilterGroup_Statics() : getFilterGroup_Dynamics(rigidId, isKinematic);
	}

	EHD bool groupFiltering(const Bp::FilterGroup::Enum group0, const Bp::FilterGroup::Enum group1, const bool*  lut)
	{
/*		const int g0 = group0 & ~3;
		const int g1 = group1 & ~3;
		if(g0==g1)
			return false;*/
		if(group0==group1)
		{
			E_ASSERT((group0 & ~BP_FILTERING_TYPE_MASK)==(group1 & ~BP_FILTERING_TYPE_MASK));
			return false;
		}

		const int type0 = group0 & BP_FILTERING_TYPE_MASK;
		const int type1 = group1 & BP_FILTERING_TYPE_MASK;
		return lut[type0*Bp::FilterType::COUNT+type1];
	}

	class BpFilter
	{
		public:
									EHDV BpFilter(bool discardKineKine, bool discardStaticKine);
									EHDV ~BpFilter();

		EHD	const bool*	getLUT()	const	{ return &mLUT[0][0];	}

						bool		mLUT[Bp::FilterType::COUNT][Bp::FilterType::COUNT];
	};
	}
}

// ===== 원본: physx/source/lowlevelaabb/src/BpFiltering.cpp
using namespace px;
using namespace Bp;

EHDI BpFilter::BpFilter(bool discardKineKine, bool discardStaticKine)
{
	for(int j = 0; j < Bp::FilterType::COUNT; j++)
		for(int i = 0; i < Bp::FilterType::COUNT; i++)
			mLUT[j][i] = false;

	mLUT[Bp::FilterType::STATIC][Bp::FilterType::DYNAMIC] = mLUT[Bp::FilterType::DYNAMIC][Bp::FilterType::STATIC] = true;
	mLUT[Bp::FilterType::STATIC][Bp::FilterType::KINEMATIC] = mLUT[Bp::FilterType::KINEMATIC][Bp::FilterType::STATIC] = !discardStaticKine;
	mLUT[Bp::FilterType::DYNAMIC][Bp::FilterType::KINEMATIC] = mLUT[Bp::FilterType::KINEMATIC][Bp::FilterType::DYNAMIC] = true;

	mLUT[Bp::FilterType::DYNAMIC][Bp::FilterType::DYNAMIC] = true;
	mLUT[Bp::FilterType::KINEMATIC][Bp::FilterType::KINEMATIC] = !discardKineKine;

	mLUT[Bp::FilterType::STATIC][Bp::FilterType::AGGREGATE] = mLUT[Bp::FilterType::AGGREGATE][Bp::FilterType::STATIC] = true;
	mLUT[Bp::FilterType::KINEMATIC][Bp::FilterType::AGGREGATE] = mLUT[Bp::FilterType::AGGREGATE][Bp::FilterType::KINEMATIC] = true;
	mLUT[Bp::FilterType::DYNAMIC][Bp::FilterType::AGGREGATE] = mLUT[Bp::FilterType::AGGREGATE][Bp::FilterType::DYNAMIC] = true;
	mLUT[Bp::FilterType::AGGREGATE][Bp::FilterType::AGGREGATE] = true;

	//Enable deformable surface interactions
	mLUT[Bp::FilterType::DEFORMABLE_SURFACE][Bp::FilterType::DYNAMIC] = mLUT[Bp::FilterType::DYNAMIC][Bp::FilterType::DEFORMABLE_SURFACE] = true;
	mLUT[Bp::FilterType::DEFORMABLE_SURFACE][Bp::FilterType::STATIC] = mLUT[Bp::FilterType::STATIC][Bp::FilterType::DEFORMABLE_SURFACE] = true;
	mLUT[Bp::FilterType::DEFORMABLE_SURFACE][Bp::FilterType::KINEMATIC] = mLUT[Bp::FilterType::KINEMATIC][Bp::FilterType::DEFORMABLE_SURFACE] = true;
	mLUT[Bp::FilterType::DEFORMABLE_SURFACE][Bp::FilterType::DEFORMABLE_SURFACE] = true;

	//Enable deformable volume interactions
	mLUT[Bp::FilterType::DEFORMABLE_VOLUME][Bp::FilterType::DYNAMIC] = mLUT[Bp::FilterType::DYNAMIC][Bp::FilterType::DEFORMABLE_VOLUME] = true;
	mLUT[Bp::FilterType::DEFORMABLE_VOLUME][Bp::FilterType::STATIC] = mLUT[Bp::FilterType::STATIC][Bp::FilterType::DEFORMABLE_VOLUME] = true;
	mLUT[Bp::FilterType::DEFORMABLE_VOLUME][Bp::FilterType::KINEMATIC] = mLUT[Bp::FilterType::KINEMATIC][Bp::FilterType::DEFORMABLE_VOLUME] = true;
	mLUT[Bp::FilterType::DEFORMABLE_VOLUME][Bp::FilterType::DEFORMABLE_VOLUME] = true;

	//Enable particle system interactions
	mLUT[Bp::FilterType::PARTICLESYSTEM][Bp::FilterType::DYNAMIC] = mLUT[Bp::FilterType::DYNAMIC][Bp::FilterType::PARTICLESYSTEM] = true;
	mLUT[Bp::FilterType::PARTICLESYSTEM][Bp::FilterType::STATIC] = mLUT[Bp::FilterType::STATIC][Bp::FilterType::PARTICLESYSTEM] = true;
	mLUT[Bp::FilterType::PARTICLESYSTEM][Bp::FilterType::KINEMATIC] = mLUT[Bp::FilterType::KINEMATIC][Bp::FilterType::PARTICLESYSTEM] = true;
	mLUT[Bp::FilterType::PARTICLESYSTEM][Bp::FilterType::PARTICLESYSTEM] = true;
}

EHDI BpFilter::~BpFilter()
{
}

// ===== 원본: physx/source/lowlevelaabb/include/BpBroadPhaseUpdate.h
namespace px
{
namespace Bp
{
typedef PxU32 ShapeHandle;
typedef PxU32 BpHandle;
#define BP_INVALID_BP_HANDLE	0x3fffffff

class BroadPhase;

class BroadPhaseUpdateData
{
public:

     /**
	 \brief A structure detailing the changes to the collection of aabbs, whose overlaps are computed in the broadphase.
	 The structure consists of per-object arrays of object bounds and object groups, and three arrays that index
	 into the per-object arrays, denoting the bounds which are to be created, updated and removed in the broad phase.

	 * each entry in the object arrays represents the same shape or aggregate from frame to frame.
	 * each entry in an index array must be less than the capacity of the per-object arrays.
	 * no index value may appear in more than one index array, and may not occur more than once in that array.

	 An index value is said to be "in use" if it has appeared in a created list in a previous update, and has not
	 since occurred in a removed list.

	 \param[in] created an array of indices describing the bounds that must be inserted into the broadphase.
	 Each index in the array must not be in use.

	 \param[in] updated an array of indices (referencing the boxBounds and boxGroups arrays) describing the bounds
	 that have moved since the last broadphase update. Each index in the array must be in use, and each object
	 whose index is in use and whose AABB has changed must appear in the update list.

	 \param[in] removed an array of indices describing the bounds that must be removed from the broad phase. Each index in
	 the array must be in use.

	 \param[in] boxBounds an array of bounds coordinates for the AABBs to be processed by the broadphase.

	 An entry is valid if its values are integer bitwise representations of floating point numbers that satisfy max>min in each dimension,
	 along with a further rule that minima(maxima) must have even(odd) values. 

	 Each entry whose index is either in use or appears in the created array must be valid. An entry whose index is either not in use or
	 appears in the removed array need not be valid.

	 \param[in]  boxGroups an array of group ids, one for each bound, used for pair filtering.  Bounds with the same group id will not be
	 reported as overlap pairs by the broad phase.  Zero is reserved for static bounds.

	 Entries in this array are immutable: the only way to change the group of an object is to remove it from the broad phase and reinsert
	 it at a different index (recall that each index must appear at most once in the created/updated/removed lists).

	 \param[in]  boxesCapacity the length of the boxBounds and boxGroups arrays.

	 \see BroadPhase::update
	 */
	EHDI BroadPhaseUpdateData(
		const ShapeHandle* created, PxU32 createdSize, 
		const ShapeHandle* updated, PxU32 updatedSize, 
		const ShapeHandle* removed, PxU32 removedSize, 
		const PxBounds3* boxBounds, const Bp::FilterGroup::Enum* boxGroups, const PxReal* boxContactDistances, const PxU32* boxEnvIDs, PxU32 boxesCapacity,
		const BpFilter& filter,
		bool stateChanged,
		bool gpuStateChanged
	) :
		mCreated		(created),
		mCreatedSize	(createdSize),
		mUpdated		(updated),
		mUpdatedSize	(updatedSize),
		mRemoved		(removed),
		mRemovedSize	(removedSize),
		mBoxBounds		(boxBounds),
		mBoxGroups		(boxGroups),
		mBoxDistances	(boxContactDistances),
		mBoxEnvIDs		(boxEnvIDs),
		mBoxesCapacity	(boxesCapacity),
		mFilter			(filter),
		mStateChanged	(stateChanged),
		mGpuStateChanged(gpuStateChanged)
	{
	}

	EHDI BroadPhaseUpdateData(
		const ShapeHandle* created, PxU32 createdSize, 
		const ShapeHandle* updated, PxU32 updatedSize, 
		const ShapeHandle* removed, PxU32 removedSize, 
		const PxBounds3* boxBounds, const Bp::FilterGroup::Enum* boxGroups, const PxReal* boxContactDistances, PxU32 boxesCapacity,
		const BpFilter& filter,
		bool stateChanged,
		bool gpuStateChanged
	) :
		mCreated		(created),
		mCreatedSize	(createdSize),
		mUpdated		(updated),
		mUpdatedSize	(updatedSize),
		mRemoved		(removed),
		mRemovedSize	(removedSize),
		mBoxBounds		(boxBounds),
		mBoxGroups		(boxGroups),
		mBoxDistances	(boxContactDistances),
		mBoxEnvIDs		(NULL),
		mBoxesCapacity	(boxesCapacity),
		mFilter			(filter),
		mStateChanged	(stateChanged),
		mGpuStateChanged(gpuStateChanged)
	{
	}

	EHDI BroadPhaseUpdateData(const BroadPhaseUpdateData& other) :
		mCreated		(other.mCreated),
		mCreatedSize	(other.mCreatedSize),
		mUpdated		(other.mUpdated),
		mUpdatedSize	(other.mUpdatedSize),
		mRemoved		(other.mRemoved),
		mRemovedSize	(other.mRemovedSize),
		mBoxBounds		(other.mBoxBounds),
		mBoxGroups		(other.mBoxGroups),
		mBoxDistances	(other.mBoxDistances),
		mBoxEnvIDs		(other.mBoxEnvIDs),
		mBoxesCapacity	(other.mBoxesCapacity),
		mFilter			(other.mFilter),
		mStateChanged	(other.mStateChanged),
		mGpuStateChanged(other.mGpuStateChanged)
	{
	}

	BroadPhaseUpdateData& operator=(const BroadPhaseUpdateData& other);

	EHD	const ShapeHandle*				getCreatedHandles()		const { return mCreated;			}
	EHD	PxU32							getNumCreatedHandles()	const { return mCreatedSize;		}

	EHD	const ShapeHandle*				getUpdatedHandles()		const { return mUpdated;			}
	EHD	PxU32							getNumUpdatedHandles()	const { return mUpdatedSize;		}

	EHD	const ShapeHandle*				getRemovedHandles()		const { return mRemoved;			}
	EHD	PxU32							getNumRemovedHandles()	const { return mRemovedSize;		}

	EHD	const PxBounds3*				getAABBs()				const { return mBoxBounds;			}
	EHD	const Bp::FilterGroup::Enum*	getGroups()				const { return mBoxGroups;			}
	EHD	const PxReal*					getContactDistance()	const { return mBoxDistances;		}
	EHD	const PxU32*					getEnvIDs()				const { return mBoxEnvIDs;			}
	EHD	PxU32							getCapacity()			const { return mBoxesCapacity;		}

	EHD	const BpFilter&					getFilter()				const { return mFilter;				}

	EHD	bool							getStateChanged()		const { return mStateChanged;		}
	EHD	bool							getGpuStateChanged()	const { return mGpuStateChanged;	}


private:

	const ShapeHandle*				mCreated;
	const PxU32						mCreatedSize;

	const ShapeHandle*				mUpdated;
	const PxU32						mUpdatedSize;

	const ShapeHandle*				mRemoved;
	const PxU32						mRemovedSize;

	const PxBounds3*				mBoxBounds;
	const Bp::FilterGroup::Enum*	mBoxGroups;
	const PxReal*					mBoxDistances;
	const PxU32*					mBoxEnvIDs;
	const PxU32						mBoxesCapacity;

	const BpFilter&					mFilter;

	const bool						mStateChanged;
	const bool						mGpuStateChanged;
};

} //namespace Bp

} //namespace px

// ===== 원본: physx/source/lowlevelaabb/include/BpBroadPhase.h
namespace px
{
	class PxcScratchAllocator;
	class PxBaseTask;

namespace Bp
{

class BroadPhaseUpdateData;

/**
\brief Base broad phase class.  Functions only relevant to MBP.
*/
class BroadPhaseBase : public PxBroadPhaseRegions, public PxUserAllocated
{
	public:
				EHDI BroadPhaseBase()	{}
	EHDI virtual		~BroadPhaseBase()	{}

	/**
	\brief Gets broad-phase caps.

	\param	caps	[out] Broad-phase caps
	*/
	EHDI virtual	void	getCaps(PxBroadPhaseCaps& caps)	const 
	{
		caps.mMaxNbRegions = 0;
	}

	// PxBroadPhaseRegions
	EHDI virtual	PxU32			getNbRegions()														const	EPX_OVERRIDE	{ return 0;			}
	EHDI virtual	PxU32			getRegions(PxBroadPhaseRegionInfo*, PxU32, PxU32)					const	EPX_OVERRIDE	{ return 0;			}
	EHDI virtual	PxU32			addRegion(const PxBroadPhaseRegion&, bool, const PxBounds3*, const PxReal*)	EPX_OVERRIDE	{ return 0xffffffff;}
	EHDI virtual	bool			removeRegion(PxU32)															EPX_OVERRIDE	{ return false;		}
	EHDI virtual	PxU32			getNbOutOfBoundsObjects()											const	EPX_OVERRIDE	{ return 0;			}
	EHDI virtual	const PxU32*	getOutOfBoundsObjects()												const	EPX_OVERRIDE	{ return NULL;		}
	//~PxBroadPhaseRegions
};

/*
\brief Structure used to report created and deleted broadphase pairs
\note The indices mVolA and mVolB correspond to the bounds indices 
BroadPhaseUpdateData::mCreated used by BroadPhase::update
\see BroadPhase::getCreatedPairs, BroadPhase::getDeletedPairs
*/
struct BroadPhasePair
{
	EHDI BroadPhasePair(ShapeHandle volA, ShapeHandle volB) :
		mVolA		(PxMin(volA, volB)),
		mVolB		(PxMax(volA, volB))
	{
	}
	EHDI BroadPhasePair() :
		mVolA		(BP_INVALID_BP_HANDLE),
		mVolB		(BP_INVALID_BP_HANDLE)
	{
	}

	ShapeHandle		mVolA;		// NB: mVolA < mVolB
	ShapeHandle		mVolB;
};

class BroadPhase : public BroadPhaseBase
{
public:

	/**
	\brief Instantiate a BroadPhase instance.
	\param[in] bpType - the bp type (either mbp or sap).  This is typically specified in PxSceneDesc. 
	\param[in] maxNbRegions is the expected maximum number of broad-phase regions.
	\param[in] maxNbBroadPhaseOverlaps is the expected maximum number of broad-phase overlaps.
	\param[in] maxNbStaticShapes is the expected maximum number of static shapes.
	\param[in] maxNbDynamicShapes is the expected maximum number of dynamic shapes.
	\param[in] contextID is the context ID parameter sent to the profiler
	\return The instantiated BroadPhase.
	\note maxNbRegions is only used if mbp is the chosen broadphase (PxBroadPhaseType::eMBP)
	\note maxNbRegions, maxNbBroadPhaseOverlaps, maxNbStaticShapes and maxNbDynamicShapes are typically specified in PxSceneLimits
	*/
	EHDV static BroadPhase* create(
		const PxBroadPhaseType::Enum bpType,
		const PxU32 maxNbRegions,
		const PxU32 maxNbBroadPhaseOverlaps,
		const PxU32 maxNbStaticShapes,
		const PxU32 maxNbDynamicShapes,
		PxU64 contextID);

	EHDV virtual	PxBroadPhaseType::Enum	getType() const = 0;

	/**
	\brief Shutdown of the broadphase.
	*/
	EHDV virtual	void	release() = 0;

	/**
	\brief Updates the broadphase and computes the lists of created/deleted pairs.

	\param[in] scratchAllocator - a PxcScratchAllocator instance used for temporary memory allocations.
	This must be non-null.

	\param[in] updateData a description of changes to the collection of aabbs since the last broadphase update.
	The changes detail the indices of the bounds that have been added/updated/removed as well as an array of all
	bound coordinates and an array of group ids used to filter pairs with the same id.
	\see BroadPhaseUpdateData

	\param[in] continuation the task that is in the queue to be executed immediately after the broadphase has completed its update. NULL is not supported.

	\note In EPX_CHECKED and EPX_DEBUG build configurations illegal input data (that does not conform to the BroadPhaseUpdateData specifications) triggers 
	a special code-path that entirely bypasses the broadphase and issues a warning message to the error stream.  No guarantees can be made about the 
	correctness/consistency of broadphase behavior with illegal input data in EPX_RELEASE and EPX_PROFILE configs because validity checks are not active 
	in these builds.
	*/
	EHDV virtual	void	update(PxcScratchAllocator* scratchAllocator, const BroadPhaseUpdateData& updateData, px::PxBaseTask* continuation) = 0;

	/**
	\brief prepare broad phase data.
	*/
	EHDV virtual	void	preBroadPhase(const Bp::BroadPhaseUpdateData& updateData) = 0;

	/**
	\brief Fetch the results of any asynchronous broad phase work.
	*/
	EHDV virtual	void	fetchBroadPhaseResults() = 0;

	/*
	\brief	Get created pairs.

	Note that each overlap pair is reported only on the frame when the overlap first occurs. The overlap persists
	until the pair appears in the list of deleted pairs or either of the bounds in the pair is removed from the broadphase.  
	A created overlap must involve at least one of the bounds of the overlap pair appearing in either the created or updated list.
	It is impossible for the same pair to appear simultaneously in the list of created and deleted overlap pairs.
	An overlap is defined as a pair of bounds that overlap on all three axes; that is when maxA > minB and maxB > minA for all three axes. 

	\param	nbCreatedPairs [out]	The number of created aabb overlap pairs computed in the execution of update() that has just completed.
	\return	The array of created aabb overlap pairs computed in the execution of update() that has just completed.
	*/
	EHDV virtual	const BroadPhasePair*	getCreatedPairs(PxU32& nbCreatedPairs)	const	= 0;

	/**
	\brief	Get deleted pairs.

	Note that a deleted pair can only be reported if that pair has already appeared in the list of created pairs in an earlier update.
	A lost overlap occurs when a pair of bounds previously overlapped on all three axes but have now separated on at least one axis.
	A lost overlap must involve at least one of the bounds of the lost overlap pair appearing in the updated list.
	Lost overlaps arising from removal of bounds from the broadphase do not appear in the list of deleted pairs.
	It is impossible for the same pair to appear simultaneously in the list of created and deleted pairs.

	\param	nbDeletedPairs [out]	The number of deleted overlap pairs computed in the execution of update() that has just completed.
	\return	The array of deleted overlap pairs computed in the execution of update() that has just completed.
	*/
	EHDV virtual	const BroadPhasePair*	getDeletedPairs(PxU32& nbDeletedPairs)	const	= 0;

	/**
	\brief After the broadphase has completed its update() function and the created/deleted pairs have been queried
	with getCreatedPairs/getDeletedPairs it is desirable to free any memory that was temporarily acquired for the update but is 
	is no longer required post-update.  This can be achieved with the function freeBuffers().
	*/
	EHDV virtual	void	freeBuffers()	= 0;

	/**
	\brief Adjust internal structures after all bounds have been adjusted due to a scene origin shift.
	*/
	EHDV virtual void	shiftOrigin(const PxVec3& shift, const PxBounds3* boundsArray, const PxReal* contactDistances) = 0;

};

} //namespace Bp

} //namespace px

// ===== 원본: physx/source/lowlevelaabb/src/BpBroadPhaseIntegerAABB.h
namespace px
{
namespace Bp
{

	/*
	\brief Encode a single float value with lossless encoding to integer
	*/
	EHD PxU32 encodeFloat(PxU32 ir)
	{
		//we may need to check on -0 and 0
		//But it should make no practical difference.
		if(ir & EPX_SIGN_BITMASK) //negative?
			return ~ir;//reverse sequence of negative numbers
		else
			return ir | EPX_SIGN_BITMASK; // flip sign
	}

	/*
	\brief Encode a single float value with lossless encoding to integer
	*/
	EHD PxU32 decodeFloat(PxU32 ir)
	{
		if(ir & EPX_SIGN_BITMASK) //positive?
			return ir & ~EPX_SIGN_BITMASK; //flip sign
		else
			return ~ir; //undo reversal
	}


/**
\brief Integer representation of PxBounds3 used by BroadPhase
\see BroadPhaseUpdateData
*/

typedef PxU32 ValType;

class IntegerAABB
{
public:

	enum
	{
		MIN_X = 0,
		MIN_Y,
		MIN_Z,
		MAX_X,
		MAX_Y,
		MAX_Z
	};

	EHDI IntegerAABB(const PxBounds3& b, PxReal contactDistance)
	{
		const PxVec3 dist(contactDistance);
		encode(PxBounds3(b.minimum - dist, b.maximum + dist));
	}

	/*
	\brief Return the minimum along a specified axis
	\param[in] i is the axis
	*/
	EHD ValType	getMin(PxU32 i)	const	{	return (mMinMax)[MIN_X+i];	}

	/*
	\brief Return the maximum along a specified axis
	\param[in] i is the axis
	*/
	EHD ValType	getMax(PxU32 i)	const	{	return (mMinMax)[MAX_X+i];	}

	/*
	\brief Return one of the six min/max values of the bound 
	\param[in] isMax determines whether a min or max value is returned
	\param[in] index is the axis 
	*/
	EHD ValType	getExtent(PxU32 isMax, PxU32 index) const
	{
		E_ASSERT(isMax<=1);
		return (mMinMax)[3*isMax+index];
	}

	/*
	\brief Return the minimum on the x axis
	*/
	EHD ValType getMinX() const { return mMinMax[MIN_X]; }

	/*
	\brief Return the minimum on the y axis
	*/
	EHD ValType getMinY() const { return mMinMax[MIN_Y]; }

	/*
	\brief Return the minimum on the z axis
	*/
	EHD ValType getMinZ() const { return mMinMax[MIN_Z]; }

	/*
	\brief Return the maximum on the x axis
	*/
	EHD ValType getMaxX() const { return mMinMax[MAX_X]; }

	/*
	\brief Return the maximum on the y axis
	*/
	EHD ValType getMaxY() const { return mMinMax[MAX_Y]; }

	/*
	\brief Return the maximum on the z axis
	*/
	EHD ValType getMaxZ() const { return mMinMax[MAX_Z]; }

	/*
	\brief Encode float bounds so they are stored as integer bounds
	\param[in] bounds is the bounds to be encoded 
	\note The integer values of minima are always even, while the integer values of maxima are always odd
	\note The encoding process masks off the last four bits for minima and masks on the last four bits for maxima.
	This keeps the bounds constant when its shape is subjected to small global pose perturbations.  In turn, this helps 
	reduce computational effort in the broadphase update by reducing the amount of sorting required on near-stationary 
	bodies that are aligned along one or more axis.
	\see decode
	*/
	EHD	void encode(const PxBounds3& bounds)
	{
		const PxU32*  min = PxUnionCast<const PxU32*, const PxF32*>(&bounds.minimum.x);
		const PxU32*  max = PxUnionCast<const PxU32*, const PxF32*>(&bounds.maximum.x);
		//Avoid min=max by enforcing the rule that mins are even and maxs are odd.
		mMinMax[MIN_X] = encodeFloatMin(min[0]);
		mMinMax[MIN_Y] = encodeFloatMin(min[1]);
		mMinMax[MIN_Z] = encodeFloatMin(min[2]);
		mMinMax[MAX_X] = encodeFloatMax(max[0]) | (1<<2);
		mMinMax[MAX_Y] = encodeFloatMax(max[1]) | (1<<2);
		mMinMax[MAX_Z] = encodeFloatMax(max[2]) | (1<<2);
	}

	/*
	\brief Decode from integer bounds to float bounds	
	\param[out] bounds is the decoded float bounds
	\note Encode followed by decode will produce a float bound larger than the original
	due to the masking in encode.
	\see encode
	*/
	EHD	void decode(PxBounds3& bounds)	const
	{
		PxU32*  min = PxUnionCast<PxU32*, PxF32*>(&bounds.minimum.x);
		PxU32*  max = PxUnionCast<PxU32*, PxF32*>(&bounds.maximum.x);
		min[0] = decodeFloat(mMinMax[MIN_X]);
		min[1] = decodeFloat(mMinMax[MIN_Y]);
		min[2] = decodeFloat(mMinMax[MIN_Z]);
		max[0] = decodeFloat(mMinMax[MAX_X]);
		max[1] = decodeFloat(mMinMax[MAX_Y]);
		max[2] = decodeFloat(mMinMax[MAX_Z]);
	}

	/*
	\brief Encode a single minimum value from integer bounds to float bounds	
	\note The encoding process masks off the last four bits for minima
	\see encode
	*/
	static EHD ValType encodeFloatMin(PxU32 source)
	{
		return ((encodeFloat(source) >> eGRID_SNAP_VAL) - 1) << eGRID_SNAP_VAL;
	}

	/*
	\brief Encode a single maximum value from integer bounds to float bounds	
	\note The encoding process masks on the last four bits for maxima
	\see encode
	*/
	static EHD ValType encodeFloatMax(PxU32 source)
	{
		return ((encodeFloat(source) >> eGRID_SNAP_VAL) + 1) << eGRID_SNAP_VAL;
	}

	/*
	\brief Shift the encoded bounds by a specified vector
	\param[in] shift is the vector used to shift the bounds
	*/
	EHD void shift(const PxVec3& shift)
	{
		px::PxBounds3 elemBounds;
		decode(elemBounds);
		elemBounds.minimum -= shift;
		elemBounds.maximum -= shift;
		encode(elemBounds);
	}

	/*
	\brief Test if this aabb lies entirely inside another aabb
	\param[in] box is the other box
	\return True if this aabb lies entirely inside box
	*/
	EHDI bool isInside(const IntegerAABB& box) const
	{
		if(box.mMinMax[MIN_X]>mMinMax[MIN_X])	return false;
		if(box.mMinMax[MIN_Y]>mMinMax[MIN_Y])	return false;
		if(box.mMinMax[MIN_Z]>mMinMax[MIN_Z])	return false;
		if(box.mMinMax[MAX_X]<mMinMax[MAX_X])	return false;
		if(box.mMinMax[MAX_Y]<mMinMax[MAX_Y])	return false;
		if(box.mMinMax[MAX_Z]<mMinMax[MAX_Z])	return false;
		return true;
	}

	/*
	\brief Test if this aabb and another intersect
	\param[in] b is the other box
	\return True if this aabb and b intersect
	*/
	EHD bool intersects(const IntegerAABB& b) const
	{
		return !(b.mMinMax[MIN_X] > mMinMax[MAX_X] || mMinMax[MIN_X] > b.mMinMax[MAX_X] ||
			b.mMinMax[MIN_Y] > mMinMax[MAX_Y] || mMinMax[MIN_Y] > b.mMinMax[MAX_Y] ||
			b.mMinMax[MIN_Z] > mMinMax[MAX_Z] || mMinMax[MIN_Z] > b.mMinMax[MAX_Z]);
	}

	EHD bool intersects1D(const IntegerAABB& b, const PxU32 axis) const
	{
		const PxU32 maxAxis = axis + 3;
		return !(b.mMinMax[axis] > mMinMax[maxAxis] || mMinMax[axis] > b.mMinMax[maxAxis]);
	}


	/*
	\brief Expand bounds to include another
	\note This is used to compute the aggregate bounds of multiple shape bounds
	\param[in] b is the bounds to be included
	*/
	EHD void include(const IntegerAABB& b)
	{
		mMinMax[MIN_X] = PxMin(mMinMax[MIN_X], b.mMinMax[MIN_X]);
		mMinMax[MIN_Y] = PxMin(mMinMax[MIN_Y], b.mMinMax[MIN_Y]);
		mMinMax[MIN_Z] = PxMin(mMinMax[MIN_Z], b.mMinMax[MIN_Z]);
		mMinMax[MAX_X] = PxMax(mMinMax[MAX_X], b.mMinMax[MAX_X]);
		mMinMax[MAX_Y] = PxMax(mMinMax[MAX_Y], b.mMinMax[MAX_Y]);
		mMinMax[MAX_Z] = PxMax(mMinMax[MAX_Z], b.mMinMax[MAX_Z]);
	}

	/*
	\brief Set the bounds to (max, max, max), (min, min, min)
	*/
	EHDI void setEmpty()
	{
		mMinMax[MIN_X] = mMinMax[MIN_Y] = mMinMax[MIN_Z] = 0xff7fffff;  //EPX_IR(EPX_MAX_F32);
		mMinMax[MAX_X] = mMinMax[MAX_Y] = mMinMax[MAX_Z] = 0x00800000;	///EPX_IR(0.0f);
	}

	ValType mMinMax[6];

private:

	enum
	{
		eGRID_SNAP_VAL = 4
	};
};

EHD ValType encodeMin(const PxBounds3& bounds, PxU32 axis, PxReal contactDistance)
{
	const PxReal val = bounds.minimum[axis] - contactDistance;
	const PxU32 min = PxUnionCast<PxU32, PxF32>(val);
	const PxU32 m = IntegerAABB::encodeFloatMin(min);
	return m;
}

EHD ValType encodeMax(const PxBounds3& bounds, PxU32 axis, PxReal contactDistance)
{
	const PxReal val = bounds.maximum[axis] + contactDistance;
	const PxU32 max = PxUnionCast<PxU32, PxF32>(val);
	const PxU32 m = IntegerAABB::encodeFloatMax(max) | (1<<2);
	return m;
}

} //namespace Bp

} //namespace px

// ===== 원본: physx/source/lowlevelaabb/src/BpBroadPhaseShared.h
namespace px
{
namespace Bp
{
	#define	INVALID_ID		0xffffffff
	#define INVALID_USER_ID	0xffffffff

	struct InternalPair : public PxUserAllocated
	{
		EHD	PxU32	getId0()	const	{ return id0_isNew & ~EPX_SIGN_BITMASK;		}
		EHD	PxU32	getId1()	const	{ return id1_isUpdated & ~EPX_SIGN_BITMASK;	}

		EHD	PxU32	isNew()		const	{ return id0_isNew & EPX_SIGN_BITMASK;		}
		EHD	PxU32	isUpdated()	const	{ return id1_isUpdated & EPX_SIGN_BITMASK;	}

		EHD	void	setNewPair(PxU32 id0, PxU32 id1)
		{
			E_ASSERT(!(id0 & EPX_SIGN_BITMASK));
			E_ASSERT(!(id1 & EPX_SIGN_BITMASK));
			id0_isNew = id0 | EPX_SIGN_BITMASK;
			id1_isUpdated = id1;
		}

		EHD	void	setNewPair2(PxU32 id0, PxU32 id1)
		{
			E_ASSERT(!(id0 & EPX_SIGN_BITMASK));
			E_ASSERT(!(id1 & EPX_SIGN_BITMASK));
			id0_isNew = id0;
			id1_isUpdated = id1;
		}

		EHD	void	setUpdated()		{ id1_isUpdated |= EPX_SIGN_BITMASK;		}
		EHD	void	clearUpdated()		{ id1_isUpdated &= ~EPX_SIGN_BITMASK;	}
		EHD	void	clearNew()			{ id0_isNew &= ~EPX_SIGN_BITMASK;		}

		protected:
		PxU32		id0_isNew;
		PxU32		id1_isUpdated;
	};

	EHD bool	differentPair(const InternalPair& p, PxU32 id0, PxU32 id1)	{ return (id0!=p.getId0()) || (id1!=p.getId1());	}
	EHD PxU32	hash(PxU32 id0, PxU32 id1)									{ return PxComputeHash( (id0&0xffff)|(id1<<16));	}
	//EHD PxU32	hash(PxU32 id0, PxU32 id1)									{ return PxComputeHash(PxU64(id0)|(PxU64(id1)<<32)) ;	}
	EHD void	sort(PxU32& id0, PxU32& id1)								{ if(id0>id1)	PxSwap(id0, id1);					}

	class PairManagerData
	{
		public:
										EHDV PairManagerData();
										EHDV ~PairManagerData();

		EHD	PxU32			getPairIndex(const InternalPair* pair)	const
										{
											return (PxU32((size_t(pair) - size_t(mActivePairs)))/sizeof(InternalPair));
										}

		// Internal version saving hash computation
		EHD InternalPair*	findPair(PxU32 id0, PxU32 id1, PxU32 hashValue) const
										{
											if(!mHashTable)
												return NULL;	// Nothing has been allocated yet

											InternalPair*  activePairs = mActivePairs;
											const PxU32*  next = mNext;

											// Look for it in the table
											PxU32 offset = mHashTable[hashValue];
											while(offset!=INVALID_ID && differentPair(activePairs[offset], id0, id1))
											{
												E_ASSERT(activePairs[offset].getId0()!=INVALID_USER_ID);
												offset = next[offset];		// Better to have a separate array for this
											}
											if(offset==INVALID_ID)
												return NULL;
											E_ASSERT(offset<mNbActivePairs);
											// Match mActivePairs[offset] => the pair is persistent

											return &activePairs[offset];
										}

		EHD	InternalPair*	addPairInternal(PxU32 id0, PxU32 id1)
										{
											// Order the ids
											sort(id0, id1);

											const PxU32 fullHashValue = hash(id0, id1);
											PxU32 hashValue = fullHashValue & mMask;

											{
												InternalPair*  p = findPair(id0, id1, hashValue);
												if(p)
												{
													p->setUpdated();
													return p;	// Persistent pair
												}
											}

											// This is a new pair
											if(mNbActivePairs >= mHashSize)
												hashValue = growPairs(fullHashValue);

											const PxU32 pairIndex = mNbActivePairs++;

											InternalPair*  p = &mActivePairs[pairIndex];
											p->setNewPair(id0, id1);
											mNext[pairIndex] = mHashTable[hashValue];
											mHashTable[hashValue] = pairIndex;
											return p;
										}

						PxU32			mHashSize;
						PxU32			mMask;
						PxU32			mNbActivePairs;
						PxU32*			mHashTable;
						PxU32*			mNext;
						InternalPair*	mActivePairs;
						PxU32			mReservedMemory;

						EHDV void			purge();
						EHDV void			reallocPairs();
						EHDV void			shrinkMemory();
						EHDV void			reserveMemory(PxU32 memSize);
				EHDV PxU32			growPairs(PxU32 fullHashValue);
						EHDV void			removePair(PxU32 id0, PxU32 id1, PxU32 hashValue, PxU32 pairIndex);
	};

	struct AABB_Xi
	{
		EHD	AABB_Xi()	{}
		EHD	~AABB_Xi()	{}

		EHD	void	initFromFloats(const void*  minX, const void*  maxX)
		{
			mMinX = encodeFloat(*reinterpret_cast<const PxU32*>(minX));
			mMaxX = encodeFloat(*reinterpret_cast<const PxU32*>(maxX));
		}

		EHD	void	initFromPxVec4(const PxVec4& min, const PxVec4& max)
		{
			initFromFloats(&min.x, &max.x);
		}

		EHD	void	operator = (const AABB_Xi& box)
		{
			mMinX = box.mMinX;
			mMaxX = box.mMaxX;
		}

		EHD	void	initSentinel()
		{
			mMinX = 0xffffffff;
		}

		EHD bool	isSentinel()	const
		{
			return mMinX == 0xffffffff;
		}

		PxU32 mMinX;
		PxU32 mMaxX;
	};

	struct AABB_YZn
	{
		EHD	AABB_YZn()	{}
		EHD	~AABB_YZn()	{}

		EHD	void	initFromPxVec4(const PxVec4& min, const PxVec4& max)
		{
			mMinY	= -min.y;
			mMinZ	= -min.z;
			mMaxY	= max.y;
			mMaxZ	= max.z;
		}

		EHD	void	operator = (const AABB_YZn& box)
		{
			using namespace px::aos;
			V4StoreA(V4LoadA(&box.mMinY), &mMinY);
		}

		float mMinY;
		float mMinZ;
		float mMaxY;
		float mMaxZ;
	};

	struct AABB_YZr
	{
		EHD	AABB_YZr()	{}
		EHD	~AABB_YZr()	{}

		EHD	void	initFromPxVec4(const PxVec4& min, const PxVec4& max)
		{
			mMinY	= min.y;
			mMinZ	= min.z;
			mMaxY	= max.y;
			mMaxZ	= max.z;
		}

		EHD	void	operator = (const AABB_YZr& box)
		{
			using namespace px::aos;
			V4StoreA(V4LoadA(&box.mMinY), &mMinY);
		}

		float mMinY;
		float mMinZ;
		float mMaxY;
		float mMaxZ;
	};

} //namespace Bp
} //namespace px

// ===== 원본: physx/source/lowlevelaabb/src/BpBroadPhaseShared.cpp
using namespace px;
using namespace Bp;

#define MBP_ALLOC(x)	EPX_ALLOC(x, "MBP")
#define MBP_FREE(x)		E_FREE(x)

static EHD void storeDwords(PxU32* dest, PxU32 nb, PxU32 value)
{
	while(nb--)
		*dest++ = value;
}

///////////////////////////////////////////////////////////////////////////////

EHDI PairManagerData::PairManagerData() :
	mHashSize		(0),
	mMask			(0),
	mNbActivePairs	(0),
	mHashTable		(NULL),
	mNext			(NULL),
	mActivePairs	(NULL),
	mReservedMemory (0)
{
}

///////////////////////////////////////////////////////////////////////////////

EHDI PairManagerData::~PairManagerData()
{
	purge();
}

///////////////////////////////////////////////////////////////////////////////

EHDI void PairManagerData::purge()
{
	MBP_FREE(mNext);
	MBP_FREE(mActivePairs);
	MBP_FREE(mHashTable);
	mHashSize		= 0;
	mMask			= 0;
	mNbActivePairs	= 0;
}

///////////////////////////////////////////////////////////////////////////////

EHDI void PairManagerData::reallocPairs()
{
	MBP_FREE(mHashTable);
	mHashTable = reinterpret_cast<PxU32*>(MBP_ALLOC(mHashSize*sizeof(PxU32)));
	storeDwords(mHashTable, mHashSize, INVALID_ID);

	// Get some bytes for new entries
	InternalPair* newPairs	= reinterpret_cast<InternalPair*>(MBP_ALLOC(mHashSize * sizeof(InternalPair)));	E_ASSERT(newPairs);
	PxU32* newNext			= reinterpret_cast<PxU32*>(MBP_ALLOC(mHashSize * sizeof(PxU32)));		E_ASSERT(newNext);

	// Copy old data if needed
	if(mNbActivePairs)
		PxMemCopy(newPairs, mActivePairs, mNbActivePairs*sizeof(InternalPair));
	// ### check it's actually needed... probably only for pairs whose hash value was cut by the and
	// yeah, since hash(id0, id1) is a constant
	// However it might not be needed to recompute them => only less efficient but still ok
	for(PxU32 i=0;i<mNbActivePairs;i++)
	{
		const PxU32 hashValue = hash(mActivePairs[i].getId0(), mActivePairs[i].getId1()) & mMask;	// New hash value with new mask
		newNext[i] = mHashTable[hashValue];
		mHashTable[hashValue] = i;
	}

	// Delete old data
	MBP_FREE(mNext);
	MBP_FREE(mActivePairs);

	// Assign new pointer
	mActivePairs = newPairs;
	mNext = newNext;
}

///////////////////////////////////////////////////////////////////////////////

EHDI void PairManagerData::shrinkMemory()
{
	// Check correct memory against actually used memory
	const PxU32 correctHashSize = PxNextPowerOfTwo(mNbActivePairs);
	if(mHashSize==correctHashSize)
		return;

	if(mReservedMemory && correctHashSize < mReservedMemory)
		return;

	// Reduce memory used
	mHashSize = correctHashSize;
	mMask = mHashSize-1;

	reallocPairs();
}

///////////////////////////////////////////////////////////////////////////////

EHDI void PairManagerData::reserveMemory(PxU32 memSize)
{
	if(!memSize)
		return;

	if(!PxIsPowerOfTwo(memSize))
		memSize = PxNextPowerOfTwo(memSize);

	mHashSize = memSize;
	mMask = mHashSize-1;

	mReservedMemory = memSize;

	reallocPairs();
}

///////////////////////////////////////////////////////////////////////////////

 EHDI PxU32 PairManagerData::growPairs(PxU32 fullHashValue)
{
	// Get more entries
	mHashSize = PxNextPowerOfTwo(mNbActivePairs+1);
	mMask = mHashSize-1;

	reallocPairs();

	// Recompute hash value with new hash size
	return fullHashValue & mMask;
}

///////////////////////////////////////////////////////////////////////////////

EHDI void PairManagerData::removePair(PxU32 /*id0*/, PxU32 /*id1*/, PxU32 hashValue, PxU32 pairIndex)
{
	// Walk the hash table to fix mNext
	{
		PxU32 offset = mHashTable[hashValue];
		E_ASSERT(offset!=INVALID_ID);

		PxU32 previous=INVALID_ID;
		while(offset!=pairIndex)
		{
			previous = offset;
			offset = mNext[offset];
		}

		// Let us go/jump us
		if(previous!=INVALID_ID)
		{
			E_ASSERT(mNext[previous]==pairIndex);
			mNext[previous] = mNext[pairIndex];
		}
		// else we were the first
		else mHashTable[hashValue] = mNext[pairIndex];
		// we're now free to reuse mNext[pairIndex] without breaking the list
	}
	// Invalidate entry

	// Fill holes
	{
		// 1) Remove last pair
		const PxU32 lastPairIndex = mNbActivePairs-1;
		if(lastPairIndex==pairIndex)
		{
			mNbActivePairs--;
		}
		else
		{
			const InternalPair* last = &mActivePairs[lastPairIndex];
			const PxU32 lastHashValue = hash(last->getId0(), last->getId1()) & mMask;

			// Walk the hash table to fix mNext
			PxU32 offset = mHashTable[lastHashValue];
			E_ASSERT(offset!=INVALID_ID);

			PxU32 previous=INVALID_ID;
			while(offset!=lastPairIndex)
			{
				previous = offset;
				offset = mNext[offset];
			}

			// Let us go/jump us
			if(previous!=INVALID_ID)
			{
				E_ASSERT(mNext[previous]==lastPairIndex);
				mNext[previous] = mNext[lastPairIndex];
			}
			// else we were the first
			else mHashTable[lastHashValue] = mNext[lastPairIndex];
			// we're now free to reuse mNext[lastPairIndex] without breaking the list


			// Don't invalidate entry since we're going to shrink the array

			// 2) Re-insert in free slot
			mActivePairs[pairIndex] = mActivePairs[lastPairIndex];
			mNext[pairIndex] = mHashTable[lastHashValue];
			mHashTable[lastHashValue] = pairIndex;

			mNbActivePairs--;
		}
	}
}

// ===== 원본: physx/source/lowlevelaabb/src/BpBroadPhaseABP.h
#define ABP_MT2

namespace internalABP{
	class ABP;
}

namespace px
{
namespace Bp
{
	class BroadPhaseABP : public BroadPhase
	{
											E_NOCOPY(BroadPhaseABP)
		public:
											EHDV BroadPhaseABP(	PxU32 maxNbBroadPhaseOverlaps,
															PxU32 maxNbStaticShapes,
															PxU32 maxNbDynamicShapes,
															PxU64 contextID,
															bool enableMT);
		EHDV virtual								~BroadPhaseABP();

	// BroadPhase
		EHDI virtual	PxBroadPhaseType::Enum		getType()					const	EPX_OVERRIDE	EPX_FINAL	{ return mEnableMT ? PxBroadPhaseType::ePABP : PxBroadPhaseType::eABP;	}
		EHDI virtual	void						release()							EPX_OVERRIDE	EPX_FINAL	{ EPX_DELETE_THIS;														}
		EHDV virtual	void						update(PxcScratchAllocator* scratchAllocator, const BroadPhaseUpdateData& updateData, px::PxBaseTask* continuation)	EPX_OVERRIDE;
		EHDI virtual	void						preBroadPhase(const Bp::BroadPhaseUpdateData&) EPX_OVERRIDE	EPX_FINAL	{}
		EHDI virtual void						fetchBroadPhaseResults()		EPX_OVERRIDE	EPX_FINAL	{}
		EHDV virtual const BroadPhasePair*		getCreatedPairs(PxU32&)	const	EPX_OVERRIDE	EPX_FINAL;
		EHDV virtual const BroadPhasePair*		getDeletedPairs(PxU32&)	const	EPX_OVERRIDE	EPX_FINAL;
		EHDV virtual void						freeBuffers()					EPX_OVERRIDE	EPX_FINAL;
		EHDV virtual void						shiftOrigin(const PxVec3& shift, const PxBounds3* boundsArray, const PxReal* contactDistances)	EPX_OVERRIDE	EPX_FINAL;
	//~BroadPhase

		internalABP::ABP*					mABP;		// PT: TODO: aggregate
				PxU32						mNbAdded;
				PxU32						mNbUpdated;
				PxU32						mNbRemoved;
				const BpHandle*				mCreatedHandles;
				const BpHandle*				mUpdatedHandles;
				const BpHandle*				mRemovedHandles;
				PxArray<BroadPhasePair>		mCreated;
				PxArray<BroadPhasePair>		mDeleted;

				const Bp::FilterGroup::Enum*mGroups;
				const BpFilter*				mFilter;

				const PxU64					mContextID;
				const bool					mEnableMT;

				EHDV void						addObjects();
				EHDV void						removeObjects();
				EHDV void						updateObjects();
	};

} //namespace Bp

} //namespace px

// ===== 원본: physx/source/lowlevelaabb/src/BpBroadPhaseABP.cpp
using namespace px;
using namespace aos;
using namespace Bp;
using namespace Cm;

/*
PT: to try:
- prepare data: sort & compute bounds in parallel? or just MT the last loop?
- switch post update & add delayed pairs?
- MT computeCreatedDeletedPairs

- why do we set the update flag for added/removed objects?
- use timestamps instead of bits?
*/

#define ABP_MT

#define CHECKPOINT(x)
//#include <stdio.h>
//#define CHECKPOINT(x)	printf(x);

//#pragma warning (disable : 4702)
#define	CODEALIGN16		//_asm	align 16
	#define ABP_SIMD_OVERLAP

//#define ABP_BATCHING		128
#define ABP_BATCHING		256

//#define USE_ABP_BUCKETS		5000	// PT: don't use buckets below that number...
#define USE_ABP_BUCKETS		512	// PT: don't use buckets below that number...
//#define USE_ABP_BUCKETS		64	// PT: don't use buckets below that number...
	#define NB_BUCKETS	5
	// Regular version: 5 buckets a la bucket pruner (4 + cross bucket)
	// Alternative version: 4 buckets + dup objects a la MBP regions
//	#define USE_ALTERNATIVE_VERSION
	#define ABP_USE_INTEGER_XS2	// Works but questionable speedups
#define NB_SENTINELS		6

//#define RECURSE_LIMIT	20000

	typedef	PxU32	ABP_Index;

	static const bool gPrepareOverlapsFlag = true;
	static const bool gUseRegularBPKernel = false;	// false to use "version 13" in box pruning series
	static const bool gUnrollLoop = true;			// true to use "version 14" in box pruning series

	typedef	float			PosXType;
	#define	SentinelValue	FLT_MAX

	typedef	PxU32			PosXType2;
	#define	SentinelValue2	0xffffffff

namespace internalABP
{
	struct SIMD_AABB4 : public PxUserAllocated
	{
		EHD	void	initFrom2(const PxBounds3& box)
		{
			mMinX = box.minimum.x;
			mMinY = box.minimum.y;
			mMinZ = box.minimum.z;
			mMaxX = box.maximum.x;
			mMaxY = box.maximum.y;
			mMaxZ = box.maximum.z;
		}

		EHD	void	operator = (const SIMD_AABB4& box)
		{
			mMinX = box.mMinX;
			mMinY = box.mMinY;
			mMinZ = box.mMinZ;
			mMaxX = box.mMaxX;
			mMaxY = box.mMaxY;
			mMaxZ = box.mMaxZ;
		}

		EHD	void	initSentinel()
		{
			mMinX = SentinelValue;
		}

		EHD bool	isSentinel()	const
		{
			return mMinX == SentinelValue;
		}

		// PT: to be able to compute bounds easily
		PosXType mMinX;
		float mMinY;
		float mMinZ;
		PosXType mMaxX;
		float mMaxY;
		float mMaxZ;
	};

#define USE_SHARED_CLASSES
	struct SIMD_AABB_X4 : public AABB_Xi
	{
		EHD	void	initFrom(const SIMD_AABB4& box)
		{
			initFromFloats(&box.mMinX, &box.mMaxX);
		}
	};

	
	struct SIMD_AABB_YZ4 : AABB_YZn
	{
		EHD	void	initFrom(const SIMD_AABB4& box)
		{
			mMinY	= -box.mMinY;
			mMinZ	= -box.mMinZ;
			mMaxY	= box.mMaxY;
			mMaxZ	= box.mMaxZ;
		}
	}
	;

#define MBP_ALLOC(x)		EPX_ALLOC(x, "MBP")
#define MBP_ALLOC_TMP(x)	EPX_ALLOC(x, "MBP_TMP")
#define MBP_FREE(x)			E_FREE(x)

#define	INVALID_ID	0xffffffff

///////////////////////////////////////////////////////////////////////////////

#define DEFAULT_NB_ENTRIES	128

	class ABP_MM
	{
		public:
									EHDI ABP_MM() : mScratchAllocator(NULL)	{}
									EHDI ~ABP_MM()							{}

			EHDV void*					frameAlloc(PxU32 size);
			EHDV void					frameFree(void* address);

			PxcScratchAllocator*	mScratchAllocator;
	};

EHDI void* ABP_MM::frameAlloc(PxU32 size)
{
	if(mScratchAllocator)
		return mScratchAllocator->alloc(size, true);
	return EPX_ALLOC(size, "frameAlloc");
}

EHDI void ABP_MM::frameFree(void* address)
{
	if(mScratchAllocator)
		mScratchAllocator->free(address);
	else
		E_FREE(address);
}

template<class T>
EHDI static T* resizeBoxesT(PxU32 oldNbBoxes, PxU32 newNbBoxes, T* boxes)
{
	T* newBoxes = reinterpret_cast<T*>(MBP_ALLOC(sizeof(T)*newNbBoxes));	
	if(oldNbBoxes)
		PxMemCopy(newBoxes, boxes, oldNbBoxes*sizeof(T));
	MBP_FREE(boxes);
	return newBoxes;
}

	class Boxes
	{
		public:
								EHDV Boxes();
								EHDV ~Boxes();

		EHD	void	init(const Boxes& boxes){ mSize = boxes.mSize; mCapacity = boxes.mCapacity;	}
		EHD	PxU32	getSize()		const	{ return mSize;										}
		EHD	PxU32	getCapacity()	const	{ return mCapacity;									}
		EHD	bool	isFull()		const	{ return mSize==mCapacity;							}
		EHD	void	reset()					{ mSize = mCapacity = 0;							}
		EHD	PxU32	popBack()				{ return --mSize;									}

//		protected:
						PxU32	mSize;
						PxU32	mCapacity;
	};

EHDI Boxes::Boxes() :
	mSize		(0),
	mCapacity	(0)
{
}

EHDI Boxes::~Boxes()
{
	reset();
}

	class StraightBoxes : public Boxes
	{
		public:
												EHDV StraightBoxes();
												EHDV ~StraightBoxes();

						EHDV void					init(PxU32 size, PxU32 capacity, SIMD_AABB4* boxes);
						EHDV void					reset();
						EHDV PxU32					resize();
						EHDV PxU32					resize(PxU32 incoming);
						EHDV bool					allocate(PxU32 nb);

		EHD	const SIMD_AABB4*		getBoxes()		const	{ return mBoxes;	}
		EHD	SIMD_AABB4*				getBoxes()				{ return mBoxes;	}

		EHD	void					setBounds(PxU32 index, const SIMD_AABB4& box)
												{
													E_ASSERT(index<mSize);
													mBoxes[index] = box;
												}

		EHD	PxU32					pushBack(const SIMD_AABB4& box)
												{
													const PxU32 index = mSize++;
													setBounds(index, box);
													return index;
												}
		private:
						SIMD_AABB4*				mBoxes;
	};

EHDI StraightBoxes::StraightBoxes() :
	mBoxes		(NULL)
{
}

EHDI StraightBoxes::~StraightBoxes()
{
	reset();
}

EHDI void StraightBoxes::reset()
{
	EPX_DELETE_ARRAY(mBoxes);
	Boxes::reset();
}

EHDI void StraightBoxes::init(PxU32 size, PxU32 capacity, SIMD_AABB4* boxes)
{
	reset();
	mSize = size;
	mCapacity = capacity;
	mBoxes = boxes;
}

EHDI PxU32 StraightBoxes::resize()
{
	const PxU32 capacity = mCapacity;
	const PxU32 size = mSize;

//	const PxU32 newCapacity = capacity ? capacity + DEFAULT_NB_ENTRIES : DEFAULT_NB_ENTRIES;
//	const PxU32 newCapacity = capacity ? capacity*2 : DEFAULT_NB_ENTRIES;
	const PxU32 newCapacity = capacity ? capacity*2 : DEFAULT_NB_ENTRIES;
	// PT: we allocate one extra box for safe SIMD loads
	mBoxes = resizeBoxesT(size, newCapacity+1, mBoxes);
	mCapacity = newCapacity;
	return newCapacity;
}

EHDI PxU32 StraightBoxes::resize(PxU32 incoming)
{
	const PxU32 capacity = mCapacity;
	const PxU32 size = mSize;
	const PxU32 minCapacity = size + incoming;
	if(minCapacity<capacity)
		return capacity;

	PxU32 newCapacity = capacity ? capacity*2 : DEFAULT_NB_ENTRIES;
	if(newCapacity<minCapacity)
		newCapacity=minCapacity;

	// PT: we allocate one extra box for safe SIMD loads
	mBoxes = resizeBoxesT(size, newCapacity+1, mBoxes);
	mCapacity = newCapacity;
	return newCapacity;
}

EHDI bool StraightBoxes::allocate(PxU32 nb)
{
	if(nb<=mSize)
		return false;
	EPX_DELETE_ARRAY(mBoxes);
	// PT: we allocate NB_SENTINELS more boxes than necessary here so we don't need to allocate one more for SIMD-load safety
	mBoxes = EPX_NEW(SIMD_AABB4)[nb+NB_SENTINELS];
	mSize = mCapacity = nb;
	return true;
}

	class SplitBoxes : public Boxes
	{
		public:
												EHDV SplitBoxes();
												EHDV ~SplitBoxes();

						EHDV void					init(PxU32 size, PxU32 capacity, SIMD_AABB_X4* boxes_X, SIMD_AABB_YZ4* boxes_YZ);
						EHDV void					init(const SplitBoxes& boxes);
						EHDV void					reset(bool freeMemory = true);
						EHDV PxU32					resize();
						EHDV PxU32					resize(PxU32 incoming);
						EHDV bool					allocate(PxU32 nb);

		EHD	const SIMD_AABB_X4*		getBoxes_X()	const	{ return mBoxes_X;	}
		EHD	SIMD_AABB_X4*			getBoxes_X()			{ return mBoxes_X;	}
		EHD	const SIMD_AABB_YZ4*	getBoxes_YZ()	const	{ return mBoxes_YZ;	}
		EHD	SIMD_AABB_YZ4*			getBoxes_YZ()			{ return mBoxes_YZ;	}

		EHD	void					setBounds(PxU32 index, const PxVec4& min, const PxVec4& max)
												{
													E_ASSERT(index<mSize);
													mBoxes_X[index].initFromPxVec4(min, max);
													mBoxes_YZ[index].initFromPxVec4(min, max);
												}

		EHD	void					setBounds(PxU32 index, const SIMD_AABB4& box)
												{
													E_ASSERT(index<mSize);
													mBoxes_X[index].initFrom(box);
													mBoxes_YZ[index].initFrom(box);
												}

		EHD	PxU32					pushBack(const SIMD_AABB4& box)
												{
													const PxU32 index = mSize++;
													setBounds(index, box);
													return index;
												}
		private:
						SIMD_AABB_X4*			mBoxes_X;
						SIMD_AABB_YZ4*			mBoxes_YZ;
	};

EHDI SplitBoxes::SplitBoxes() :
	mBoxes_X	(NULL),
	mBoxes_YZ	(NULL)
{
}

EHDI SplitBoxes::~SplitBoxes()
{
	reset();
}

EHDI void SplitBoxes::reset(bool freeMemory)
{
	if(freeMemory)
	{
		MBP_FREE(mBoxes_YZ);
		MBP_FREE(mBoxes_X);
	}
	mBoxes_X = NULL;
	mBoxes_YZ = NULL;
	Boxes::reset();
}

EHDI void SplitBoxes::init(PxU32 size, PxU32 capacity, SIMD_AABB_X4* boxes_X, SIMD_AABB_YZ4* boxes_YZ)
{
	reset();
	mSize = size;
	mCapacity = capacity;
	mBoxes_X = boxes_X;
	mBoxes_YZ = boxes_YZ;
}

EHDI void SplitBoxes::init(const SplitBoxes& boxes)
{
	reset();
	Boxes::init(boxes);
	mBoxes_X = const_cast<SIMD_AABB_X4*>(boxes.getBoxes_X());
	mBoxes_YZ = const_cast<SIMD_AABB_YZ4*>(boxes.getBoxes_YZ());
}

EHDI PxU32 SplitBoxes::resize()
{
	const PxU32 capacity = mCapacity;
	const PxU32 size = mSize;

//	const PxU32 newCapacity = capacity ? capacity + DEFAULT_NB_ENTRIES : DEFAULT_NB_ENTRIES;
//	const PxU32 newCapacity = capacity ? capacity*2 : DEFAULT_NB_ENTRIES;
	const PxU32 newCapacity = capacity ? capacity*2 : DEFAULT_NB_ENTRIES;
	mBoxes_X = resizeBoxesT(size, newCapacity, mBoxes_X);
	mBoxes_YZ = resizeBoxesT(size, newCapacity, mBoxes_YZ);
	mCapacity = newCapacity;
	return newCapacity;
}

EHDI PxU32 SplitBoxes::resize(PxU32 incoming)
{
	const PxU32 capacity = mCapacity;
	const PxU32 size = mSize;
	const PxU32 minCapacity = size + incoming;
	if(minCapacity<capacity)
		return capacity;

	PxU32 newCapacity = capacity ? capacity*2 : DEFAULT_NB_ENTRIES;
	if(newCapacity<minCapacity)
		newCapacity=minCapacity;

	mBoxes_X = resizeBoxesT(size, newCapacity, mBoxes_X);
	mBoxes_YZ = resizeBoxesT(size, newCapacity, mBoxes_YZ);
	mCapacity = newCapacity;
	return newCapacity;
}

EHDI bool SplitBoxes::allocate(PxU32 nb)
{
	if(nb<=mSize)
		return false;
	MBP_FREE(mBoxes_YZ);
	MBP_FREE(mBoxes_X);
	mBoxes_X = reinterpret_cast<SIMD_AABB_X4*>(MBP_ALLOC(sizeof(SIMD_AABB_X4)*(nb+NB_SENTINELS)));
	mBoxes_YZ = reinterpret_cast<SIMD_AABB_YZ4*>(MBP_ALLOC(sizeof(SIMD_AABB_YZ4)*nb));
	E_ASSERT(!(size_t(mBoxes_YZ) & 15));
	mSize = mCapacity = nb;
	return true;
}

	typedef	SplitBoxes		StaticBoxes;
	typedef	SplitBoxes		DynamicBoxes;

///////////////////////////////////////////////////////////////////////////////

	struct ABP_Object : public PxUserAllocated
	{
		EHD	ABP_Object() : mIndex(INVALID_ID)
		{
		}

	private:
		PxU32		mIndex;			// Out-to-in, maps user handle to internal array. mIndex indexes either the static or dynamic array.
		// PT: the type won't be available for removed objects so we have to store it there. That uses 2 bits.
		// Then the "data" will need one more bit for marking sleeping objects so that leaves 28bits for the actual index.
		EHD	void	setData(PxU32 index, FilterType::Enum type)
		{
//			mIndex = index;
			index <<= 2;
			index |= type;
			mIndex = index;
		}
	public:
		// PT: TODO: rename "index" to data everywhere
		EHD	void	setActiveIndex(PxU32 index, FilterType::Enum type)
		{
			const PxU32 boxData = (index+index);
			setData(boxData, type);
		}

		EHD	void	setSleepingIndex(PxU32 index, FilterType::Enum type)
		{
			const PxU32 boxData = (index+index)|1;
			E_ASSERT(getType()==type);
			setData(boxData, type);
		}

		EHD		FilterType::Enum	getType()	const
		{
			return FilterType::Enum(mIndex&3);
		}

		EHD		PxU32	getData()	const
		{
			return mIndex>>2;
		}

		EHD		void	invalidateIndex()
		{
			mIndex = INVALID_ID;
		}
		EHD		bool	isValid()	const
		{
			return mIndex != INVALID_ID;
		}

	};

	typedef ABP_Object ABPEntry;

///////////////////////////////////////////////////////////////////////////////

//#define BIT_ARRAY_STACK	512

	static EHD PxU32 bitsToDwords(PxU32 nbBits)
	{
		return (nbBits>>5) + ((nbBits&31) ? 1 : 0);
	}

	// Use that one instead of an array of bools. Takes less ram, nearly as fast [no bounds checkings and so on].
	class BitArray
	{
		public:
										EHDV BitArray();
										EHDV BitArray(PxU32 nbBits);
										EHDV ~BitArray();

						EHDV bool			init(PxU32 nbBits);
						EHDV void			empty();
						EHDV void			resize(PxU32 nbBits);

		EHD	void			checkResize(PxU32 bitNumber)
										{
											const PxU32 index = bitNumber>>5;
											if(index>=mSize)
												resize(bitNumber);
										}

		EHD	void			setBitChecked(PxU32 bitNumber)
										{
											const PxU32 index = bitNumber>>5;
											if(index>=mSize)
												resize(bitNumber);
											mBits[index] |= 1<<(bitNumber&31);
										}

		EHD	void			clearBitChecked(PxU32 bitNumber)
										{
											const PxU32 index = bitNumber>>5;
											if(index>=mSize)
												resize(bitNumber);
											mBits[index] &= ~(1<<(bitNumber&31));
										}
		// Data management
		EHD	void			setBit(PxU32 bitNumber)					{ mBits[bitNumber>>5] |= 1<<(bitNumber&31);				}
		EHD	void			clearBit(PxU32 bitNumber)				{ mBits[bitNumber>>5] &= ~(1<<(bitNumber&31));			}
		EHD	void			toggleBit(PxU32 bitNumber)				{ mBits[bitNumber>>5] ^= 1<<(bitNumber&31);				}

		EHD	void			clearAll()								{ PxMemZero(mBits, mSize*4);							}
		EHD	void			setAll()								{ PxMemSet(mBits, 0xff, mSize*4);						}

		// Data access
		EHD	PxIntBool		isSet(PxU32 bitNumber)			const	{ return PxIntBool(mBits[bitNumber>>5] & (1<<(bitNumber&31)));		}
		EHD	PxIntBool		isSetChecked(PxU32 bitNumber)	const
										{
											const PxU32 index = bitNumber>>5;
											if(index>=mSize)
												return 0;
											return PxIntBool(mBits[index] & (1<<(bitNumber&31)));
										}

		EHD	const PxU32*	getBits()						const	{ return mBits;											}
		EHD	PxU32			getSize()						const	{ return mSize;											}

		protected:
						PxU32*			mBits;		//!< Array of bits
						PxU32			mSize;		//!< Size of the array in dwords
	};

///////////////////////////////////////////////////////////////////////////////

EHDI BitArray::BitArray() : mBits(NULL), mSize(0)
{
}

EHDI BitArray::BitArray(PxU32 nbBits) : mBits(NULL), mSize(0)
{
	init(nbBits);
}

EHDI BitArray::~BitArray()
{
	empty();
}

EHDI void BitArray::empty()
{
		MBP_FREE(mBits);
	mBits = NULL;
	mSize = 0;
}

EHDI bool BitArray::init(PxU32 nbBits)
{
	mSize = bitsToDwords(nbBits);
	// Get ram for n bits
		MBP_FREE(mBits);
		mBits = reinterpret_cast<PxU32*>(MBP_ALLOC(sizeof(PxU32)*mSize));

	// Set all bits to 0
	clearAll();
	return true;
}

EHDI void BitArray::resize(PxU32 nbBits)
{
	const PxU32 newSize = bitsToDwords(nbBits+128);
	PxU32* newBits = NULL;
	{
		// Old buffer was stack or allocated, new buffer is allocated
		newBits = reinterpret_cast<PxU32*>(MBP_ALLOC(sizeof(PxU32)*newSize));
		if(mSize)
			PxMemCopy(newBits, mBits, sizeof(PxU32)*mSize);
	}
	const PxU32 remain = newSize - mSize;
	if(remain)
		PxMemZero(newBits + mSize, remain*sizeof(PxU32));

		MBP_FREE(mBits);
	mBits = newBits;
	mSize = newSize;
}

///////////////////////////////////////////////////////////////////////////////

EHDI static ABP_Index* resizeMapping(PxU32 oldNbBoxes, PxU32 newNbBoxes, ABP_Index* mapping)
{
	ABP_Index* newMapping = reinterpret_cast<ABP_Index*>(MBP_ALLOC(sizeof(ABP_Index)*newNbBoxes));
	if(oldNbBoxes)
		PxMemCopy(newMapping, mapping, oldNbBoxes*sizeof(ABP_Index));
	MBP_FREE(mapping);
	return newMapping;
}

	struct ABP_Object;

	struct DelayedPair
	{
		PxU32	mID0;
		PxU32	mID1;
		PxU32	mHash;
	};

	class ABP_PairManager : public PairManagerData
	{
		public:
														EHDV ABP_PairManager();
														EHDV ~ABP_PairManager();

						EHDV InternalPair*					addPair						(PxU32 id0, PxU32 id1);
						EHDV void							computeCreatedDeletedPairs	(PxArray<BroadPhasePair>& createdPairs, PxArray<BroadPhasePair>& deletedPairs, const BitArray& updated, const BitArray& removed);
						EHDV void							addDelayedPair	(PxArray<DelayedPair>& delayedPairs, const ABP_Index* mInToOut0, const ABP_Index* mInToOut1, PxU32 index0, PxU32 index1) const;
						EHDV void							addDelayedPairs	(const PxArray<DelayedPair>& delayedPairs);
						EHDV void							addDelayedPairs2(PxArray<BroadPhasePair>& createdPairs, const PxArray<DelayedPair>& delayedPairs);
						EHDV void							resizeForNewPairs(PxU32 nbDelayedPairs);
						const Bp::FilterGroup::Enum*	mGroups;
						const ABP_Index*				mInToOut0;
						const ABP_Index*				mInToOut1;
						const bool*						mLUT;
	};

	///////////////////////////////////////////////////////////////////////////

	struct ABP_SharedData
	{
		EHD				ABP_SharedData() :
										mABP_Objects			(NULL),
										mABP_Objects_Capacity	(0)
									{
									}

						EHDV void		resize(BpHandle userID);

		EHD	void		checkResize(PxU32 maxID)
									{
										if(mABP_Objects_Capacity<maxID+1)
											resize(maxID);
									}

						ABP_Object*	mABP_Objects;
						PxU32		mABP_Objects_Capacity;
						BitArray	mUpdatedObjects;	// Indexed by ABP_ObjectIndex
						BitArray	mRemovedObjects;	// Indexed by ABP_ObjectIndex
	};

EHDI void ABP_SharedData::resize(BpHandle userID)
{
	const PxU32 oldCapacity = mABP_Objects_Capacity;

	PxU32 newCapacity = mABP_Objects_Capacity ? mABP_Objects_Capacity*2 : 256;
	if(newCapacity<userID+1)
		newCapacity = userID+1;

	ABP_Object* newObjects = EPX_NEW(ABP_Object)[newCapacity];
	if(mABP_Objects)
		PxMemCopy(newObjects, mABP_Objects, oldCapacity*sizeof(ABP_Object));
	EPX_DELETE_ARRAY(mABP_Objects);
	mABP_Objects = newObjects;
	mABP_Objects_Capacity = newCapacity;
}

	class BoxManager
	{
		public:
											EHDV BoxManager(FilterType::Enum type);
											EHDV ~BoxManager();

						EHDV void				reset();
						EHDI void				setSourceData(const PxBounds3* bounds, const float* distances)
											{
												mAABBManagerBounds = bounds;
												mAABBManagerDistances = distances;
											}

						EHDV void				addObjects(const BpHandle*  userIDs, PxU32 nb, ABP_SharedData*  sharedData);
						EHDV void				removeObject(ABPEntry& object, BpHandle userID);
						EHDV void				updateObject(ABPEntry& object, BpHandle userID);

						EHDV void				prepareData(RadixSortBuffered& rs, ABP_Object*  objects, PxU32 objectsCapacity, ABP_MM& memoryManager, PxU64 contextID);

//		EHD	PxU32				isThereWorkToDo()		const	{ return mNbUpdated;	}
		EHD	bool				isThereWorkToDo()		const	{ return mNbUpdated || mNbRemovedSleeping;	}	// PT: temp & test, maybe we do that differently in the end
		EHD	PxU32				getNbUpdatedBoxes()		const	{ return mNbUpdated;		}
		EHD	PxU32				getNbNonUpdatedBoxes()	const	{ return mNbSleeping;		}
		EHD	const DynamicBoxes&	getUpdatedBoxes()		const	{ return mUpdatedBoxes;		}
		EHD	const DynamicBoxes&	getSleepingBoxes()		const	{ return mSleepingBoxes;	}
		EHD	const ABP_Index*	getRemap_Updated()		const	{ return mInToOut_Updated;	}
		EHD	const ABP_Index*	getRemap_Sleeping()		const	{ return mInToOut_Sleeping;	}
		EHD	const PxBounds3&	getUpdatedBounds()		const	{ return mUpdatedBounds;	}

		private:
						FilterType::Enum	mType;

						// PT: refs to source data (not owned). Currently separate arrays, ideally should be merged.
						const PxBounds3*	mAABBManagerBounds;
						const float*		mAABBManagerDistances;

						// New & updated objects
						PxBounds3			mUpdatedBounds;		// Bounds around updated dynamic objects, computed in prepareData().
						ABP_Index*			mInToOut_Updated;	// Maps boxes to mABP_Objects
						PxU32				mNbUpdated;
						PxU32				mMaxNbUpdated;
						DynamicBoxes		mUpdatedBoxes;

						// Sleeping objects
						ABP_Index*			mInToOut_Sleeping;	// Maps boxes to mABP_Objects
						PxU32				mNbSleeping;
						DynamicBoxes		mSleepingBoxes;

						// Removed sleeping
						PxU32				mNbRemovedSleeping;

						EHDV void				purgeRemovedFromSleeping(ABP_Object*  objects, PxU32 objectsCapacity);
	};

EHDI BoxManager::BoxManager(FilterType::Enum type) :
	mType					(type),
	mAABBManagerBounds		(NULL),
	mAABBManagerDistances	(NULL),
	mInToOut_Updated		(NULL),
	mNbUpdated				(0),
	mMaxNbUpdated			(0),
	mInToOut_Sleeping		(NULL),
	mNbSleeping				(0),
	mNbRemovedSleeping		(0)
{
}

EHDI BoxManager::~BoxManager()
{
	reset();
}

EHDI void BoxManager::reset()
{
	mMaxNbUpdated = mNbUpdated = mNbSleeping = 0;
	E_FREE(mInToOut_Updated);
	E_FREE(mInToOut_Sleeping);
	mUpdatedBoxes.reset();
	mSleepingBoxes.reset();
}

static EHD PxU32 isNewOrUpdated(PxU32 data)
{
	return data & EPX_SIGN_BITMASK;
}

static EHD PxU32 markAsNewOrUpdated(PxU32 data)
{
	return data | EPX_SIGN_BITMASK;
}

static EHD PxU32 removeNewOrUpdatedMark(PxU32 data)
{
	return data & ~EPX_SIGN_BITMASK;
}

// BpHandle = index in main/shared arrays like mAABBManagerBounds / mAABBManagerDistances
E_STATIC_ASSERT(sizeof(BpHandle)==sizeof(ABP_Index));
EHDI void BoxManager::addObjects(const BpHandle*  userIDs, PxU32 nb, ABP_SharedData*  sharedData)
{
	// PT: we're called for each batch.

	// PT: TODO: fix the BpHandle/ABP_Index mix
	const PxU32 currentSize = mNbUpdated;
	const PxU32 currentCapacity = mMaxNbUpdated;
	const PxU32 newSize = currentSize + nb;

	ABP_Index* remap;
	if(newSize>currentCapacity)
	{
		const PxU32 minCapacity = PxMax(newSize, 1024u);
		const PxU32 newCapacity = PxMax(minCapacity, currentCapacity*2);
		E_ASSERT(newCapacity>=newSize);
		mMaxNbUpdated = newCapacity;
		remap = resizeMapping(currentSize, newCapacity, mInToOut_Updated);
	}
	else
	{
		remap = mInToOut_Updated;
	}
	mInToOut_Updated = remap;
	mNbUpdated = newSize;

	// PT: we only copy the new handles for now. The bounds will be computed later in "prepareData".
	// PT: TODO: do we even need to copy them? Can't we just reuse the source ptr directly?
	{
		E_ASSERT(currentSize+nb<=mMaxNbUpdated);
		remap += currentSize;
		PxU32 nbToGo = nb;
		while(nbToGo--)
		{
			const BpHandle userID = *userIDs++;

			E_ASSERT(!isNewOrUpdated(userID));
			*remap++ = markAsNewOrUpdated(userID);

			if(sharedData)
				sharedData->mUpdatedObjects.setBit(userID);
		}
	}
}

// PT: TODO: inline this again
EHDI void BoxManager::removeObject(ABPEntry& object, BpHandle userID)
{
	E_UNUSED(userID);
	const PxU32 boxData = object.getData();
	const PxU32 boxIndex = boxData>>1;
	if(boxData&1)
	{
		// Sleeping object.
		E_ASSERT(boxIndex<mNbSleeping);
		E_ASSERT(mInToOut_Sleeping[boxIndex]==userID);
		E_ASSERT(mInToOut_Sleeping[boxIndex] != INVALID_ID);	// PT: can that happen if we update and remove an object in the same frame or does the AABB take care of it?
		mInToOut_Sleeping[boxIndex] = INVALID_ID;
		mNbRemovedSleeping++;
		E_ASSERT(mNbRemovedSleeping<=mNbSleeping);
	}
	else
	{
		// PT: remove active object, i.e. one that was previously in "updated" arrays. 
		E_ASSERT(boxIndex<mNbUpdated);
		E_ASSERT(boxIndex<mMaxNbUpdated);
		E_ASSERT(mInToOut_Updated[boxIndex]==userID);
		E_ASSERT(mInToOut_Updated[boxIndex] != INVALID_ID);
		// PT: TODO: do we need this at all? We could use 'userID' to access the removed bitmap...
		mInToOut_Updated[boxIndex] = INVALID_ID;
	}
}

// PT: TODO: inline this again
EHDI void BoxManager::updateObject(ABPEntry& object, BpHandle userID)
{
	E_UNUSED(userID);
	const PxU32 boxData = object.getData();
	const PxU32 boxIndex = boxData>>1;
	if(boxData&1)
	{
		// PT: benchmark for this codepath: MBP.UpdateSleeping

		// Sleeping object. We must reactivate it, i.e:
		// - remove it from the array of sleeping objects
		// - add it to the array of active/updated objects

		// First we remove:
		{
			E_ASSERT(boxIndex<mNbSleeping);
			E_ASSERT(mInToOut_Sleeping[boxIndex]==userID);
			E_ASSERT(mInToOut_Sleeping[boxIndex] != INVALID_ID);
			mInToOut_Sleeping[boxIndex] = INVALID_ID;
			mNbRemovedSleeping++;
			E_ASSERT(mNbRemovedSleeping<=mNbSleeping);
		}

		// Then we add
		// PT: TODO: revisit / improve this maybe
		addObjects(&userID, 1, NULL);	// Don't pass sharedData because the bitmap has already been updated by the calling code
	}
	else
	{
		// Active object, i.e. it was updated in previous frame and it's already in mInToOut_Updated array
		E_ASSERT(boxIndex<mNbUpdated);
		E_ASSERT(boxIndex<mMaxNbUpdated);
		E_ASSERT(mInToOut_Updated[boxIndex]==userID);
		mInToOut_Updated[boxIndex] = markAsNewOrUpdated(mInToOut_Updated[boxIndex]);
	}
}


static EHD void initSentinels(SIMD_AABB_X4*  boxesX, const PxU32 size)
{
	for(PxU32 i=0;i<NB_SENTINELS;i++)
		boxesX[size+i].initSentinel();
}

EHDI void BoxManager::purgeRemovedFromSleeping(ABP_Object*  objects, PxU32 objectsCapacity)
{
	CHECKPOINT("purgeRemovedFromSleeping\n");

	E_UNUSED(objectsCapacity);
	E_ASSERT(mNbRemovedSleeping);
	E_ASSERT(mNbSleeping);

	// PT: TODO: do we need to allocate separate buffers here?

	// PT: we reach this codepath when:
	// - no object has been added or updated
	// - sleeping objects have been removed
	// So we have to purge the removed objects from the sleeping array. We cannot entirely ignore the removals since we compute collisions
	// between sleeping arrays and active arrays for bipartite cases. So we either have to remove the invalid entries immediately, or make
	// sure they don't report collisions. We could ignore collisions when the remapped ID is "INVALID_ID" but that would be an additional
	// test for each potential pair, i.e. it's a constant cost. We cannot tweak the removed bounding boxes (e.g. mark them as empty) because
	// they are sorted, and the tweak would break the sorting and the collision loop. Keeping all removed objects in the array also means
	// there is more data to parse all the time, i.e. there is a performance cost again. So for now we just remove all deleted entries here.
	// ==> also tweaking the sleeping boxes might break the "merge sleeping" array code
	E_ASSERT(mNbRemovedSleeping<=mNbSleeping);

	if(mNbRemovedSleeping==mNbSleeping)
	{
		// PT: remove everything
		mSleepingBoxes.reset();
		E_FREE(mInToOut_Sleeping);
		mNbSleeping = mNbRemovedSleeping = 0;
		return;
	}

	const PxU32 expectedTotal = mNbSleeping - mNbRemovedSleeping;
	PxU32 nbRemovedFound = 0;
	PxU32 nbSleepingLeft = 0;
	const PxU32 sleepCapacity = mSleepingBoxes.getCapacity();
	if(expectedTotal>=sleepCapacity/2)
	{	
		// PT: remove holes, keep same data buffers
		SIMD_AABB_X4* boxesX = mSleepingBoxes.getBoxes_X();
		SIMD_AABB_YZ4* boxesYZ = mSleepingBoxes.getBoxes_YZ();
		ABP_Index* remap = mInToOut_Sleeping;

		for(PxU32 i=0;i<mNbSleeping;i++)
		{
			const PxU32 boxIndex = remap[i];
			if(boxIndex==INVALID_ID)
			{
				nbRemovedFound++;
			}
			else
			{
				E_ASSERT(nbSleepingLeft<expectedTotal);
				if(i!=nbSleepingLeft)
				{
					remap[nbSleepingLeft] = boxIndex;
					boxesX[nbSleepingLeft] = boxesX[i];
					boxesYZ[nbSleepingLeft] = boxesYZ[i];
				}

				{
					E_ASSERT(boxIndex<objectsCapacity);
					objects[boxIndex].setSleepingIndex(nbSleepingLeft, mType);
				}
				nbSleepingLeft++;
			}
		}
		E_ASSERT(nbSleepingLeft==expectedTotal);
		E_ASSERT(nbSleepingLeft+nbRemovedFound==mNbSleeping);
		E_UNUSED(nbRemovedFound);

		initSentinels(boxesX, expectedTotal);

		mSleepingBoxes.mSize = expectedTotal;
	}
	else
	{
		// PT: remove holes, get fresh memory buffers
		SIMD_AABB_X4* dstBoxesX = reinterpret_cast<SIMD_AABB_X4*>(MBP_ALLOC(sizeof(SIMD_AABB_X4)*(expectedTotal+NB_SENTINELS)));
		SIMD_AABB_YZ4* dstBoxesYZ = reinterpret_cast<SIMD_AABB_YZ4*>(MBP_ALLOC(sizeof(SIMD_AABB_YZ4)*(expectedTotal+NB_SENTINELS)));
		initSentinels(dstBoxesX, expectedTotal);
		BpHandle*  dstRemap = reinterpret_cast<BpHandle*>(EPX_ALLOC(expectedTotal*sizeof(BpHandle), "tmp"));

		const SIMD_AABB_X4*  srcDataX = mSleepingBoxes.getBoxes_X();
		const SIMD_AABB_YZ4*  srcDataYZ = mSleepingBoxes.getBoxes_YZ();
		const ABP_Index*  srcRemap = mInToOut_Sleeping;

		for(PxU32 i=0;i<mNbSleeping;i++)
		{
			const PxU32 boxIndex = srcRemap[i];
			if(boxIndex==INVALID_ID)
			{
				nbRemovedFound++;
			}
			else
			{
				E_ASSERT(nbSleepingLeft<expectedTotal);
				dstRemap[nbSleepingLeft] = boxIndex;
				dstBoxesX[nbSleepingLeft] = srcDataX[i];
				dstBoxesYZ[nbSleepingLeft] = srcDataYZ[i];

				{
					E_ASSERT(boxIndex<objectsCapacity);
					objects[boxIndex].setSleepingIndex(nbSleepingLeft, mType);
				}
				nbSleepingLeft++;
			}
		}
		E_ASSERT(nbSleepingLeft==expectedTotal);
		E_ASSERT(nbSleepingLeft+nbRemovedFound==mNbSleeping);

		// PT: TODO: double check all this
		mSleepingBoxes.init(expectedTotal, expectedTotal, dstBoxesX, dstBoxesYZ);

		E_FREE(mInToOut_Sleeping);
		mInToOut_Sleeping = dstRemap;
	}
	mNbSleeping = expectedTotal;
	mNbRemovedSleeping = 0;
}

static EHD PosXType2 getNextCandidateSorted(PxU32 offsetSorted, const PxU32 nbSorted, const SIMD_AABB_X4*  sortedDataX, const PxU32*  sleepingIndices)
{
	return offsetSorted<nbSorted ? sortedDataX[sleepingIndices[offsetSorted]].mMinX : SentinelValue2;
}
static EHD PosXType2 getNextCandidateNonSorted(PxU32 offsetNonSorted, const PxU32 nbToSort, const SIMD_AABB_X4*  toSortDataX)
{
	return offsetNonSorted<nbToSort ? toSortDataX[offsetNonSorted].mMinX : SentinelValue2;
}

E_STATIC_ASSERT(sizeof(BpHandle)==sizeof(float));
EHDI void BoxManager::prepareData(RadixSortBuffered& /*rs*/, ABP_Object*  objects, PxU32 objectsCapacity, ABP_MM& memoryManager, PxU64 contextID)
{
	E_UNUSED(contextID);

	// PT: mNbUpdated = number of objects in the updated buffer, could have been updated this frame or previous frame
	const PxU32 size = mNbUpdated;
	if(!size)
	{
		if(mNbRemovedSleeping)
		{
			// PT: benchmark for this codepath: MBP.RemoveHalfSleeping
			purgeRemovedFromSleeping(objects, objectsCapacity);
		}
		return;
	}

	E_ASSERT(mAABBManagerBounds);
	E_ASSERT(mAABBManagerDistances);
	E_ASSERT(mInToOut_Updated);

	// Prepare new/updated objects
	const ABP_Index*  remap = mInToOut_Updated;
	const PxBounds3*  bounds = mAABBManagerBounds;
	const float*  distances = mAABBManagerDistances;

	float*  keys = NULL;

	// newOrUpdatedIDs: *userIDs* of objects that have been added or updated this frame.
	// sleepingIndices: *indices* (not userIDs) of non-updated objects within mInToOut_Updated
	PxU32* tempBuffer = NULL;
	PxU32* newOrUpdatedIDs = tempBuffer;
	PxU32* sleepingIndices = tempBuffer;

	// PT: mNbUpdated / mInToOut_Updated contains:
	// 1) objects added this frame (from addObject(s))
	// 2) objects updated this frame (from updateObject(s))
	// 3) objects updated the frame before, not updated this frame, i.e. they are now "sleeping"
	// 4) objects updated the frame before, then removed (from removeObject(s))
	//
	// We split the current array into separate groups:
	// - 1) & 2) go to "temp", count is "nbUpdated"
	// - 3) go to "temp2", count is "nbSleeping"
	// - 4) are filtered out. No special processing is needed because the updated data is always parsed/recreated here anyway.
	//   So if we don't actively add removed objects to the new buffers, they get removed as a side-effect.

	PxU32 nbUpdated = 0;	// PT: number of objects updated this frame
	PxU32 nbSleeping = 0;
	PxU32 nbRemoved = 0;	// PT: number of removed objects that were previously located in the udpated array

	// PT: TODO: could we do the work within mInToOut_Updated?
	// - updated objects have invalidated bounds so we don't need to preserve their order
	// - we need to preserve the order of sleeping objects to avoid re-sorting them
	// - we cannot use MTF since it breaks the order
	// - parse backward and move sleeping objects to the back? but then we might have to move the sleeping boxes at the same time

	for(PxU32 i=0;i<size;i++)
	{
		E_ASSERT(i<mMaxNbUpdated);
		const PxU32 index = remap[i];
		if(index==INVALID_ID)
		{
			nbRemoved++;
		}
		else
		{
			if(!tempBuffer)
			{
				tempBuffer = reinterpret_cast<PxU32*>(memoryManager.frameAlloc(size*sizeof(PxU32)));
				newOrUpdatedIDs = tempBuffer;
				sleepingIndices = tempBuffer;
			}

			if(isNewOrUpdated(index))
			{
				// PT: new or updated object
				if(!keys)
					keys = reinterpret_cast<float*>(EPX_ALLOC(size*sizeof(float), "tmp"));

				// PT: in this version we compute the key on-the-fly, i.e. it will be computed twice overall. We could make this
				// faster by merging bounds and distances inside the AABB manager.
				const BpHandle userID = removeNewOrUpdatedMark(index);

				keys[nbUpdated] = bounds[userID].minimum.x - distances[userID];

				newOrUpdatedIDs[size - 1 - nbUpdated] = userID;
				nbUpdated++;
			}
			else
			{
				// PT: sleeping object
				sleepingIndices[nbSleeping++] = i;
			}
		}
	}
	E_ASSERT(nbRemoved + nbUpdated + nbSleeping == size);
	E_UNUSED(nbRemoved);

	// PT: we must process the sleeping objects first, because the bounds of new sleeping objects are located in the existing updated buffers.
	// PT: TODO: *HOWEVER* we could sort things right now and then reuse the "keys" buffer?

	if(nbSleeping)
	{
		// PT: must merge these guys to current sleeping array
		// They should already be in sorted order and we should already have the boxes.

		if(mNbSleeping)
		{
			// PT: benchmark for this codepath: MBP.MergeSleeping
			CHECKPOINT("Merging sleeping objects\n");

			// PT: here, we need to merge two arrays of sleeping objects together:
			// - the ones already contained inside mSleepingBoxes
			// - the new sleeping objects currently contained in mUpdatedBoxes
			// Both of them should already be sorted.
			// PT: TODO: super subtle stuff going on there, to revisit

			// PT: TODO: revisit names
			PxU32 offsetSorted = 0;
			const PxU32 nbSorted = nbSleeping;
			const SIMD_AABB_X4*  sortedDataX = mUpdatedBoxes.getBoxes_X();
			const SIMD_AABB_YZ4*  sortedDataYZ = mUpdatedBoxes.getBoxes_YZ();
			const ABP_Index*  sortedRemap = mInToOut_Updated;

			PxU32 offsetNonSorted = 0;
			const PxU32 nbToSort = mNbSleeping;
			const SIMD_AABB_X4*  toSortDataX = mSleepingBoxes.getBoxes_X();
			const SIMD_AABB_YZ4*  toSortDataYZ = mSleepingBoxes.getBoxes_YZ();
			const ABP_Index*  toSortRemap = mInToOut_Sleeping;

			E_ASSERT(mNbRemovedSleeping<=mNbSleeping);
			PosXType2 nextCandidateNonSorted = getNextCandidateNonSorted(offsetNonSorted, nbToSort, toSortDataX);

			PosXType2 nextCandidateSorted = getNextCandidateSorted(offsetSorted, nbSorted, sortedDataX, sleepingIndices);

			const PxU32 nbTotal = nbSorted + nbToSort - mNbRemovedSleeping;

			SIMD_AABB_X4* dstBoxesX = reinterpret_cast<SIMD_AABB_X4*>(MBP_ALLOC(sizeof(SIMD_AABB_X4)*(nbTotal+NB_SENTINELS)));
			SIMD_AABB_YZ4* dstBoxesYZ = reinterpret_cast<SIMD_AABB_YZ4*>(MBP_ALLOC(sizeof(SIMD_AABB_YZ4)*(nbTotal+NB_SENTINELS)));

			initSentinels(dstBoxesX, nbTotal);
			BpHandle*  dstRemap = reinterpret_cast<BpHandle*>(EPX_ALLOC(nbTotal*sizeof(BpHandle), "tmp"));

			PxU32 i=0;
			PxU32 nbToGo = nbSorted + nbToSort;
			while(nbToGo--)
			{
				PxU32 boxIndex;
				{
					if(nextCandidateNonSorted<nextCandidateSorted)
					{
						boxIndex = toSortRemap[offsetNonSorted];
						if(boxIndex!=INVALID_ID)
						{
							dstRemap[i] = boxIndex;
							dstBoxesX[i] = toSortDataX[offsetNonSorted];
							dstBoxesYZ[i] = toSortDataYZ[offsetNonSorted];
						}

						offsetNonSorted++;

						nextCandidateNonSorted = getNextCandidateNonSorted(offsetNonSorted, nbToSort, toSortDataX);
					}
					else
					{
						const PxU32 j = sleepingIndices[offsetSorted];
						E_ASSERT(j<size);
						boxIndex = sortedRemap[j];
						E_ASSERT(boxIndex!=INVALID_ID);
							dstRemap[i] = boxIndex;
							dstBoxesX[i] = sortedDataX[j];
							dstBoxesYZ[i] = sortedDataYZ[j];

						offsetSorted++;

						nextCandidateSorted = getNextCandidateSorted(offsetSorted, nbSorted, sortedDataX, sleepingIndices);
					}
				}

				if(boxIndex!=INVALID_ID)
				{
					E_ASSERT(boxIndex<objectsCapacity);
					objects[boxIndex].setSleepingIndex(i, mType);
					i++;
				}
			}
			E_ASSERT(i==nbTotal);
			E_ASSERT(offsetSorted+offsetNonSorted==nbSorted+nbToSort);

			// PT: TODO: double check all this
			mSleepingBoxes.init(nbTotal, nbTotal, dstBoxesX, dstBoxesYZ);

			E_FREE(mInToOut_Sleeping);
			mInToOut_Sleeping = dstRemap;
			mNbSleeping = nbTotal;
			mNbRemovedSleeping = 0;
		}
		else
		{
			// PT: benchmark for this codepath: MBP.ActiveToSleeping
			CHECKPOINT("Active objects become sleeping objects\n");

			// PT: TODO: optimize allocs
			BpHandle* inToOut_Sleeping;
			if(mSleepingBoxes.allocate(nbSleeping))
			{
				inToOut_Sleeping = reinterpret_cast<BpHandle*>(EPX_ALLOC(nbSleeping*sizeof(BpHandle), "tmp"));
				E_FREE(mInToOut_Sleeping);
				mInToOut_Sleeping = inToOut_Sleeping;
			}
			else
			{
				inToOut_Sleeping = mInToOut_Sleeping;
			}

			const SIMD_AABB_X4* srcBoxesX = mUpdatedBoxes.getBoxes_X();
			const SIMD_AABB_YZ4* srcBoxesYZ = mUpdatedBoxes.getBoxes_YZ();
			SIMD_AABB_X4* dstBoxesX = mSleepingBoxes.getBoxes_X();
			SIMD_AABB_YZ4* dstBoxesYZ = mSleepingBoxes.getBoxes_YZ();
			initSentinels(dstBoxesX, nbSleeping);
			for(PxU32 ii=0;ii<nbSleeping;ii++)
			{
				const PxU32 i = sleepingIndices[ii];	// PT: TODO: remove this indirection
				const PxU32 index = remap[i];
				E_ASSERT(index!=INVALID_ID);
				inToOut_Sleeping[ii] = index;
				dstBoxesX[ii] = srcBoxesX[i];
				dstBoxesYZ[ii] = srcBoxesYZ[i];
				{
					E_ASSERT(index<objectsCapacity);
					objects[index].setSleepingIndex(ii, mType);
				}
			}
			mNbSleeping = nbSleeping;
		}
	}
	else
	{
		// PT: no sleeping objects in updated buffer
		if(mNbSleeping)
		{
			if(mNbRemovedSleeping)
			{
				// PT: benchmark for this codepath: MBP.UpdateSleeping
				purgeRemovedFromSleeping(objects, objectsCapacity);
			}
		}
		else
		{
			E_ASSERT(!mNbRemovedSleeping);
		}
	}

	if(nbUpdated)
	{
		// PT: benchmark for this codepath: MBP.Update64KObjects
		CHECKPOINT("Create updated objects\n");

		// PT: we need to sort here because we reuse the "keys" buffer just afterwards
		PxU32* ranks0 = reinterpret_cast<PxU32*>(memoryManager.frameAlloc(sizeof(PxU32)*nbUpdated));
		PxU32* ranks1 = reinterpret_cast<PxU32*>(memoryManager.frameAlloc(sizeof(PxU32)*nbUpdated));

		StackRadixSort(rs, ranks0, ranks1);
		const PxU32* sorted;
		{
			EPX_PROFILE_ZONE("Sort", contextID);
			sorted = rs.Sort(keys, nbUpdated).GetRanks();
		}

		// PT:
		// - shuffle the remap table, store it in sorted order (we can probably use the "recyclable" array here again)
		// - compute bounds on-the-fly, store them in sorted order

		// PT: TODO: the "keys" array can be much bigger than stricly necessary here
		BpHandle* inToOut_Updated_Sorted;
		if(mUpdatedBoxes.allocate(nbUpdated))
		{
			inToOut_Updated_Sorted = reinterpret_cast<BpHandle*>(keys);

			E_FREE(mInToOut_Updated);
			mInToOut_Updated = inToOut_Updated_Sorted;
		}
		else
		{
			E_FREE(keys);

			inToOut_Updated_Sorted = mInToOut_Updated;
		}
		SIMD_AABB_X4*  dstBoxesX = mUpdatedBoxes.getBoxes_X();
		initSentinels(dstBoxesX, nbUpdated);

		Vec4V minV = V4Load(FLT_MAX);
		Vec4V maxV = V4Load(-FLT_MAX);
		// PT: TODO: parallel? Everything indexed by i should be fine, things indexed by userID might have some false sharing
		for(PxU32 i=0;i<nbUpdated;i++)
		{
			const PxU32 sortedIndex = *sorted++;

			const BpHandle userID = newOrUpdatedIDs[size - 1 - sortedIndex];
			E_ASSERT(i<size);
			inToOut_Updated_Sorted[i] = userID;

			{
				E_ASSERT(userID<objectsCapacity);
				objects[userID].setActiveIndex(i, mType);
			}
	
			// PT: TODO: refactor with computeMBPBounds?
			{
				const PxBounds3& b = bounds[userID];
				const Vec4V contactDistanceV = V4Load(distances[userID]);
				const Vec4V inflatedMinV = V4Sub(V4LoadU(&b.minimum.x), contactDistanceV);
				const Vec4V inflatedMaxV = V4Add(V4LoadU(&b.maximum.x), contactDistanceV);	// PT: this one is safe because we allocated one more box in the array (in BoundsArray::initEntry)
				minV = V4Min(minV, inflatedMinV);
				maxV = V4Max(maxV, inflatedMaxV);
				// PT: TODO better
				PxVec4 boxMin;
				PxVec4 boxMax;
				V4StoreA(inflatedMinV, &boxMin.x);
				V4StoreA(inflatedMaxV, &boxMax.x);

				mUpdatedBoxes.setBounds(i, boxMin, boxMax);
			}
		}
		StoreBounds(mUpdatedBounds, minV, maxV)
		memoryManager.frameFree(ranks1);
		memoryManager.frameFree(ranks0);
	}
	else
	{
		// PT: benchmark for this codepath: MBP.MergeSleeping / MBP.Remove64KObjects
		CHECKPOINT("Free updated objects\n");

		E_FREE(keys);

		mUpdatedBoxes.reset();
		E_FREE(mInToOut_Updated);
	}
	mNbUpdated = mMaxNbUpdated = nbUpdated;

	if(tempBuffer)
		memoryManager.frameFree(tempBuffer);
}

namespace
{
	struct PairManagerMT
	{
		const ABP_PairManager*	mSharedPM;
		PxArray<DelayedPair>	mDelayedPairs;

		const ABP_Index*		mInToOut0;
		const ABP_Index*		mInToOut1;

		//char					mBuffer[256];
	};
}

static EHD void outputPair(PairManagerMT& pairManager, PxU32 index0, PxU32 index1)
{
	pairManager.mSharedPM->addDelayedPair(pairManager.mDelayedPairs, pairManager.mInToOut0, pairManager.mInToOut1, index0, index1);
}

	#define NB_BIP_TASKS	15
	enum ABP_TaskID
	{
		ABP_TASK_0,
		ABP_TASK_1,
	};

	class ABP_InternalTask : public PxLightCpuTask
	{
		public:
							EHDI ABP_InternalTask(ABP_TaskID id) : mBP(NULL), mID(id)	{}
		EHDI virtual	const char* getName()	const	EPX_OVERRIDE
		{
			return "ABP_InternalTask";
		}

		EHDV virtual void run()	EPX_OVERRIDE;

		EHDI virtual bool	isHighPriority()	const	EPX_OVERRIDE	{ return true; }

		BroadPhaseABP*			mBP;
		ABP_TaskID				mID;
	};

	class ABP_CompleteBoxPruningStartTask;

	class ABP_CompleteBoxPruningTask : public PxLightCpuTask
	{
	public:
							EHDI ABP_CompleteBoxPruningTask() :
								mStartTask(NULL),
								mType(0),
								mID(0)
							{
							}

		EHDI virtual	const char* getName()	const	EPX_OVERRIDE
		{
			return "ABP_CompleteBoxPruningTask";
		}

		EHDV virtual void run()	EPX_OVERRIDE;

		EHDI virtual bool	isHighPriority()	const	EPX_OVERRIDE	{ return true; }

		ABP_CompleteBoxPruningStartTask*	mStartTask;

		PxU16					mType;
		PxU16					mID;

		PxU32					mCounter;
		const SIMD_AABB_X4*		mBoxListX;
		const SIMD_AABB_YZ4*	mBoxListYZ;
		const PxU32*			mRemap;

		PxU32					mCounter4;
		const SIMD_AABB_X4*		mBoxListX4;
		const SIMD_AABB_YZ4*	mBoxListYZ4;
		const PxU32*			mRemap4;

		PairManagerMT			mPairs;

		EHD	bool	isThereWorkToDo()	const
		{
			if(!mCounter)
				return false;

			if(mType)
				return mCounter4!=0;

			return true;
		}
	};

	class ABP_CompleteBoxPruningEndTask : public PxLightCpuTask
	{
		public:
							EHDI ABP_CompleteBoxPruningEndTask() : mStartTask(NULL)	{}

		EHDI virtual	const char* getName()	const	EPX_OVERRIDE
		{
			return "ABP_CompleteBoxPruningEndTask";
		}

		EHDV virtual void run()	EPX_OVERRIDE;

		EHDI virtual bool	isHighPriority()	const	EPX_OVERRIDE	{ return true; }

		ABP_CompleteBoxPruningStartTask*	mStartTask;
	};

	class ABP_CompleteBoxPruningStartTask : public PxLightCpuTask
	{
		public:
							EHDV ABP_CompleteBoxPruningStartTask();

		EHDI virtual	const char* getName()	const	EPX_OVERRIDE
		{
			return "ABP_CompleteBoxPruningStartTask";
		}

		EHDV void	setup(
			//ABP_MM& memoryManager,
			const PxBounds3& updatedBounds,
			ABP_PairManager*  pairManager,
			PxU32 nb,
			const SIMD_AABB_X4*  listX,
			const SIMD_AABB_YZ4*  listYZ,
			const ABP_Index*  inputRemap,
			PxU64 contextID);

		EHDV void	addDelayedPairs();
		EHDV void	addDelayedPairs2(PxArray<BroadPhasePair>& createdPairs);
		
		EHDV virtual void run()	EPX_OVERRIDE;

		EHDI virtual bool	isHighPriority()	const	EPX_OVERRIDE	{ return true; }

		const SIMD_AABB_X4*				mListX;
		const SIMD_AABB_YZ4*			mListYZ;
		const ABP_Index*				mInputRemap;
		ABP_PairManager*				mPairManager;

		PxU32*							mRemap;
		SIMD_AABB_X4*					mBoxListXBuffer;
		SIMD_AABB_YZ4*					mBoxListYZBuffer;
		PxU32							mCounters[NB_BUCKETS];
		SIMD_AABB_X4*					mBoxListX[NB_BUCKETS];
		SIMD_AABB_YZ4*					mBoxListYZ[NB_BUCKETS];
		PxU32*							mRemapBase[NB_BUCKETS];
		PxBounds3						mBounds;
		PxU32							mNb;

		ABP_CompleteBoxPruningTask		mTasks[9];
		ABP_CompleteBoxPruningEndTask	mEndTask;
	};

	typedef	BoxManager	DynamicManager;
	typedef	BoxManager	StaticManager;

	class ABP : public PxUserAllocated
	{
												E_NOCOPY(ABP)
		public:
												EHDV ABP(PxU64 contextID);
												EHDV ~ABP();

						EHDV void					preallocate(PxU32 nbObjects, PxU32 maxNbOverlaps);
						EHDV void					reset();
						EHDV void					freeBuffers();

						EHDV void					addStaticObjects(const BpHandle* userIDs, PxU32 nb, PxU32 maxID);
						EHDV void					addDynamicObjects(const BpHandle* userIDs, PxU32 nb, PxU32 maxID);
						EHDV void					addKinematicObjects(const BpHandle* userIDs, PxU32 nb, PxU32 maxID);
						EHDV void					removeObject(BpHandle userID);
						EHDV void					updateObject(BpHandle userID);
						EHDV void					findOverlaps(PxBaseTask* continuation, const Bp::FilterGroup::Enum*  groups, const bool*  lut);
						EHDV PxU32					finalize(PxArray<BroadPhasePair>& createdPairs, PxArray<BroadPhasePair>& deletedPairs);
						EHDV void					shiftOrigin(const PxVec3& shift, const PxBounds3* boundsArray, const PxReal* contactDistances);

						EHDV void					setTransientData(const PxBounds3* bounds, const PxReal* contactDistance);

						EHDV void					Region_prepareOverlaps();

						ABP_MM					mMM;
						BoxManager				mSBM;
						DynamicManager			mDBM;
						RadixSortBuffered		mRS;
						DynamicManager			mKBM;
						ABP_SharedData			mShared;
						ABP_PairManager			mPairManager;
				const	PxU64					mContextID;
						ABP_InternalTask		mTask0;
						ABP_InternalTask		mTask1;
				ABP_CompleteBoxPruningStartTask	mCompleteBoxPruningTask0;
				ABP_CompleteBoxPruningStartTask	mCompleteBoxPruningTask1;
					ABP_CompleteBoxPruningTask	mBipTasks[NB_BIP_TASKS];

						EHDV void					addDelayedPairs();
						EHDV void					addDelayedPairs2(PxArray<BroadPhasePair>& createdPairs);
	};

	#define ABP_OVERLAP_TEST(x)	SIMD_OVERLAP_TEST(x)

///////////////////////////////////////////////////////////////////////////////

EHDI ABP_PairManager::ABP_PairManager() :
	mGroups		(NULL),
	mInToOut0	(NULL),
	mInToOut1	(NULL),
	mLUT		(NULL)
{
}

///////////////////////////////////////////////////////////////////////////////

EHDI ABP_PairManager::~ABP_PairManager()
{
}

///////////////////////////////////////////////////////////////////////////////

EHDI InternalPair* ABP_PairManager::addPair(PxU32 index0, PxU32 index1)
{
	const PxU32 id0 = mInToOut0[index0];
	const PxU32 id1 = mInToOut1[index1];
	E_ASSERT(id0!=id1);
	E_ASSERT(id0!=INVALID_ID);
	E_ASSERT(id1!=INVALID_ID);
	E_ASSERT(mGroups);

	{
		if(!groupFiltering(mGroups[id0], mGroups[id1], mLUT))
			return NULL;
	}

	return addPairInternal(id0, id1);
}

EHDI void ABP_PairManager::addDelayedPair(PxArray<DelayedPair>& delayedPairs, const ABP_Index* inToOut0, const ABP_Index* inToOut1, PxU32 index0, PxU32 index1) const
{
	/*const*/ PxU32 id0 = inToOut0[index0];
	/*const*/ PxU32 id1 = inToOut1[index1];
	E_ASSERT(id0!=id1);
	E_ASSERT(id0!=INVALID_ID);
	E_ASSERT(id1!=INVALID_ID);
	E_ASSERT(mGroups);

	{
		if(!groupFiltering(mGroups[id0], mGroups[id1], mLUT))
			return;
	}

	if(1)
	{
		// Order the ids
		sort(id0, id1);

		const PxU32 fullHashValue = hash(id0, id1);
		PxU32 hashValue = fullHashValue & mMask;

		{
			InternalPair* /**/ p = findPair(id0, id1, hashValue);
			if(p)
			{
				p->setUpdated();	// ### PT: potential false sharing here
				//return p;	// Persistent pair
				return;	// Persistent pair
			}
		}

	{
		/*// This is a new pair
		if(mNbActivePairs >= mHashSize)
			hashValue = growPairs(fullHashValue);

		const PxU32 pairIndex = mNbActivePairs++;

		InternalPair*  p = &mActivePairs[pairIndex];
		p->setNewPair(id0, id1);
		mNext[pairIndex] = mHashTable[hashValue];
		mHashTable[hashValue] = pairIndex;
		return p;*/

		DelayedPair* newPair = Cm::reserveContainerMemory(delayedPairs, 1);
		newPair->mID0 = id0;
		newPair->mID1 = id1;
		newPair->mHash = fullHashValue;
	}
	}
}

EHDI void ABP_PairManager::resizeForNewPairs(PxU32 nbDelayedPairs)
{
	PxU32 currentNbPairs = mNbActivePairs;

	const PxU32 newNbPairs = currentNbPairs + nbDelayedPairs;

	const PxU32 newHashSize = PxNextPowerOfTwo(newNbPairs + 1);
	if(newHashSize == mHashSize)
		return;

	// Get more entries
	mHashSize = newHashSize;
	mMask = newHashSize - 1;

	//reallocPairs();
	{
		MBP_FREE(mHashTable);
		mHashTable = reinterpret_cast<PxU32*>(MBP_ALLOC(mHashSize*sizeof(PxU32)));
		//storeDwords(mHashTable, mHashSize, INVALID_ID);
		if(0)
		{
			PxU32 nb = mHashSize;
			PxU32* dest = mHashTable;
			while(nb--)
				*dest++ = INVALID_ID;
		}
		else
			PxMemSet(mHashTable, 0xff, mHashSize*sizeof(PxU32));

		// Get some bytes for new entries
		InternalPair* newPairs	= reinterpret_cast<InternalPair*>(MBP_ALLOC(mHashSize * sizeof(InternalPair)));	E_ASSERT(newPairs);
		PxU32* newNext			= reinterpret_cast<PxU32*>(MBP_ALLOC(mHashSize * sizeof(PxU32)));				E_ASSERT(newNext);

		// Copy old data if needed
		if(currentNbPairs)
			PxMemCopy(newPairs, mActivePairs, currentNbPairs*sizeof(InternalPair));
		// ### check it's actually needed... probably only for pairs whose hash value was cut by the and
		// yeah, since hash(id0, id1) is a constant
		// However it might not be needed to recompute them => only less efficient but still ok

		// PT: TODO: in heavy scenes like Avalanche100K the number of pairs gets close to a million, and this loop becomes very expensive. Revisit.
		for(PxU32 i=0;i<currentNbPairs;i++)
		{
			const PxU32 hashValue = hash(mActivePairs[i].getId0(), mActivePairs[i].getId1()) & mMask;	// New hash value with new mask
			newNext[i] = mHashTable[hashValue];
			mHashTable[hashValue] = i;
		}

		// Delete old data
		MBP_FREE(mNext);
		MBP_FREE(mActivePairs);

		// Assign new pointer
		mActivePairs = newPairs;
		mNext = newNext;
	}
}

EHDI void ABP_PairManager::addDelayedPairs(const PxArray<DelayedPair>& delayedPairs)
{
	if(0)
	{
		PxU32 nbDelayedPairs = delayedPairs.size();
		const DelayedPair* pairs = delayedPairs.begin();
		while(nbDelayedPairs--)
		{
			const DelayedPair& dp = *pairs++;

			const PxU32 fullHashValue = dp.mHash;
			PxU32 hashValue = fullHashValue & mMask;

			if(mNbActivePairs >= mHashSize)
				hashValue = growPairs(fullHashValue);

			const PxU32 pairIndex = mNbActivePairs++;

			InternalPair*  p = &mActivePairs[pairIndex];
			p->setNewPair(dp.mID0, dp.mID1);
			mNext[pairIndex] = mHashTable[hashValue];
			mHashTable[hashValue] = pairIndex;
		}
	}
	else
	{
		PxU32 nbDelayedPairs = delayedPairs.size();
		PxU32 currentNbPairs = mNbActivePairs;
		//resizeForNewPairs(nbDelayedPairs);

		{
			const PxU32 mask = mMask;
			PxU32*  hashTable = mHashTable;
			PxU32*  next = mNext;
			InternalPair*  internalPairs = mActivePairs;
			const DelayedPair*  pairs = delayedPairs.begin();
			while(nbDelayedPairs--)
			{
				const DelayedPair& dp = *pairs++;

				const PxU32 fullHashValue = dp.mHash;
				const PxU32 hashValue = fullHashValue & mask;

				E_ASSERT(currentNbPairs < mHashSize);

				const PxU32 pairIndex = currentNbPairs++;

				internalPairs[pairIndex].setNewPair(dp.mID0, dp.mID1);
				next[pairIndex] = hashTable[hashValue];
				hashTable[hashValue] = pairIndex;
			}
			mNbActivePairs = currentNbPairs;
		}
	}
}

EHDI void ABP_PairManager::addDelayedPairs2(PxArray<BroadPhasePair>& createdPairs, const PxArray<DelayedPair>& delayedPairs)
{
	PxU32 nbDelayedPairs = delayedPairs.size();
	PxU32 currentNbPairs = mNbActivePairs;
	//resizeForNewPairs(nbDelayedPairs);

	BroadPhasePair* newPair = Cm::reserveContainerMemory(createdPairs, nbDelayedPairs);

	{
		const PxU32 mask = mMask;
		PxU32*  hashTable = mHashTable;
		PxU32*  next = mNext;
		InternalPair*  internalPairs = mActivePairs;
		const DelayedPair*  pairs = delayedPairs.begin();
		while(nbDelayedPairs--)
		{
			const DelayedPair& dp = *pairs++;

			const PxU32 fullHashValue = dp.mHash;
			const PxU32 hashValue = fullHashValue & mask;

			E_ASSERT(currentNbPairs < mHashSize);

			const PxU32 pairIndex = currentNbPairs++;

			internalPairs[pairIndex].setNewPair2(dp.mID0, dp.mID1);
			{
				newPair->mVolA = dp.mID0;
				newPair->mVolB = dp.mID1;
				newPair++;
			}

			next[pairIndex] = hashTable[hashValue];
			hashTable[hashValue] = pairIndex;
		}
		mNbActivePairs = currentNbPairs;
	}
}

///////////////////////////////////////////////////////////////////////////////

	#define SIMD_OVERLAP_TEST_14a(box)	em_movemask_ps(em_cmpngt_ps(b, em_load_ps(box)))==15

	#define SIMD_OVERLAP_INIT_9c(box)	\
			em128 b = em_shuffle_ps(em_load_ps(&box.mMinY), em_load_ps(&box.mMinY), 78);\
			const float Coeff = -1.0f;\
			b = em_mul_ps(b, em_load1_ps(&Coeff));

	#define SIMD_OVERLAP_TEST_9c(box)					\
			const em128 a = em_load_ps(&box.mMinY);	\
			const em128 d = em_cmpge_ps(a, b);		\
			if(em_movemask_ps(d)==15)

	#define SIMD_OVERLAP_PRELOAD_BOX0	SIMD_OVERLAP_INIT_9c(box0)
	#define SIMD_OVERLAP_TEST(x)		SIMD_OVERLAP_TEST_9c(x)


static EHD void outputPair(ABP_PairManager& pairManager, PxU32 index0, PxU32 index1)
{
	pairManager.addPair(index0, index1);
}

template<const int codepath, class ABP_PairManagerT>
EHDI static void boxPruningKernel(	PxU32 nb0, PxU32 nb1,
								const SIMD_AABB_X4*  boxes0_X, const SIMD_AABB_X4*  boxes1_X,
								const SIMD_AABB_YZ4*  boxes0_YZ, const SIMD_AABB_YZ4*  boxes1_YZ,
								const ABP_Index*  inToOut0, const ABP_Index*  inToOut1,
								ABP_PairManagerT*  pairManager)
{
	pairManager->mInToOut0 = inToOut0;
	pairManager->mInToOut1 = inToOut1;

	PxU32 index0 = 0;
	PxU32 runningIndex1 = 0;

	while(runningIndex1<nb1 && index0<nb0)
	{
		const SIMD_AABB_X4& box0_X = boxes0_X[index0];
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

		const SIMD_AABB_YZ4& box0 = boxes0_YZ[index0];
		SIMD_OVERLAP_PRELOAD_BOX0

		if(gUseRegularBPKernel)
		{
			PxU32 index1 = runningIndex1;

			while(boxes1_X[index1].mMinX<=maxLimit)
			{
				ABP_OVERLAP_TEST(boxes1_YZ[index1])
				{
					outputPair(*pairManager, index0, index1);
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
						outputPair(*pairManager, index0, Index1);
					}
					Offset += 8;
				}
			}
			else
			{
#define BIP_VERSION4
	#define BLOCK4(x, label)	{const float* box = reinterpret_cast<const float*>(CurrentBoxListYZ + Offset*2 + x*2);	\
								if(SIMD_OVERLAP_TEST_14a(box))															\
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
			outputPair(*pairManager, index0, Index1);
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
						goto OverlapFound;																\
					Offset += 8;

		goto LoopStart;
		CODEALIGN16
OverlapFound:
		{
			const PxU32 Index1 = PxU32(CurrentBoxListX + Offset - reinterpret_cast<const char*>(boxes1_X))>>3;
			outputPair(*pairManager, index0, Index1);
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

template<class ABP_PairManagerT>
EHDI static /*EHD*/ void doBipartiteBoxPruning_Leaf(
		ABP_PairManagerT*  pairManager,
		PxU32 nb0,
		PxU32 nb1,
		const SIMD_AABB_X4*  boxes0_X,
		const SIMD_AABB_X4*  boxes1_X,
		const SIMD_AABB_YZ4*  boxes0_YZ,
		const SIMD_AABB_YZ4*  boxes1_YZ,
		const ABP_Index*  remap0,
		const ABP_Index*  remap1
		)
{
	E_ASSERT(boxes0_X[nb0].isSentinel());
	E_ASSERT(boxes1_X[nb1].isSentinel());
	boxPruningKernel<0>(nb0, nb1, boxes0_X, boxes1_X, boxes0_YZ, boxes1_YZ, remap0, remap1, pairManager);
	boxPruningKernel<1>(nb1, nb0, boxes1_X, boxes0_X, boxes1_YZ, boxes0_YZ, remap1, remap0, pairManager);
}

template<class ABP_PairManagerT>
static EHD void doBipartiteBoxPruning_Leaf(ABP_PairManagerT*  pairManager,
		PxU32 nb0, PxU32 nb1, const SplitBoxes& boxes0, const SplitBoxes& boxes1, const ABP_Index*  remap0, const ABP_Index*  remap1)
{
	doBipartiteBoxPruning_Leaf(pairManager, nb0, nb1, boxes0.getBoxes_X(), boxes1.getBoxes_X(), boxes0.getBoxes_YZ(), boxes1.getBoxes_YZ(), remap0, remap1);
}

template<class ABP_PairManagerT>
EHDI static void doCompleteBoxPruning_Leaf(	ABP_PairManagerT*  pairManager, PxU32 nb,
										const SIMD_AABB_X4*  boxes_X,
										const SIMD_AABB_YZ4*  boxes_YZ,
										const ABP_Index*  remap)
{
	pairManager->mInToOut0 = remap;
	pairManager->mInToOut1 = remap;

	PxU32 index0 = 0;
	PxU32 runningIndex = 0;
	while(runningIndex<nb && index0<nb)
	{
		const SIMD_AABB_X4& box0_X = boxes_X[index0];
		const PosXType2 maxLimit = box0_X.mMaxX;

		const PosXType2 minLimit = box0_X.mMinX;
		while(boxes_X[runningIndex++].mMinX<minLimit);

		const SIMD_AABB_YZ4& box0 = boxes_YZ[index0];
		SIMD_OVERLAP_PRELOAD_BOX0

		if(gUseRegularBPKernel)
		{
			PxU32 index1 = runningIndex;
			while(boxes_X[index1].mMinX<=maxLimit)
			{
				ABP_OVERLAP_TEST(boxes_YZ[index1])
				{
					outputPair(*pairManager, index0, index1);
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
						outputPair(*pairManager, index0, Index);
					}
					Offset += 8;
				}
			}
			else
			{
#define VERSION4c
#define VERSION3	// Enable this as our safe loop
	#define BLOCK4(x, label)	{const float* box = reinterpret_cast<const float*>(CurrentBoxListYZ + Offset*2 + x*2);	\
							if(SIMD_OVERLAP_TEST_14a(box))																\
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
			outputPair(*pairManager, index0, Index);
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
						goto BeforeLoop;																\
					Offset += 8;

		goto StartLoop;
		CODEALIGN16
BeforeLoop:
		{
			const PxU32 Index = PxU32(CurrentBoxListX + Offset - reinterpret_cast<const char*>(boxes_X))>>3;
			outputPair(*pairManager, index0, Index);
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

E_GTABLE(PxU8, gCodes, 16, { 4, 4, 4, 255, 4, 3, 2, 255, 4, 1, 0, 255, 255, 255, 255, 255 })

static EHD PxU8 classifyBoxNew(const SIMD_AABB_YZ4& boxYZ, const float limitY, const float limitZ)
{
	// PT: mins have been negated for SIMD tests
	const bool upperPart = (-boxYZ.mMinZ) > limitZ;
	const bool rightPart = (-boxYZ.mMinY) > limitY;
	const bool lowerPart = boxYZ.mMaxZ < limitZ;
	const bool leftPart = boxYZ.mMaxY < limitY;

	// Table-based box classification avoids many branches
	const PxU32 Code = PxU32(rightPart)|(PxU32(leftPart)<<1)|(PxU32(upperPart)<<2)|(PxU32(lowerPart)<<3);
	E_ASSERT(gCodes_p()[Code]!=255);
	return gCodes_p()[Code];
}


EHDI void ABP_CompleteBoxPruningTask::run()
{
//	printf("Running ABP_CompleteBoxPruningTask\n");

	//printf("ABP_Task_%d - thread ID %d\n", mID, PxU32(PxThread::getId()));
	//printf("Count: %d\n", mCounter);

	bool runComplete = false;
	bool runBipartite = false;

	if(mType==0)
		runComplete = true;
	else
		runBipartite = true;

	if(runComplete)
		doCompleteBoxPruning_Leaf(&mPairs, mCounter, mBoxListX, mBoxListYZ, mRemap);

	if(runBipartite)
		doBipartiteBoxPruning_Leaf(&mPairs,
									mCounter, mCounter4,
									mBoxListX, mBoxListX4,
									mBoxListYZ, mBoxListYZ4,
									mRemap, mRemap4);
}

EHDI void ABP_CompleteBoxPruningEndTask::run()
{
//	printf("Running ABP_CompleteBoxPruningEndTask\n");

	//memoryManager.frameFree(Remap);
	//memoryManager.frameFree(BoxListYZBuffer);
	//memoryManager.frameFree(BoxListXBuffer);
	// PT: TODO: revisit allocs
	E_FREE(mStartTask->mRemap);
	E_FREE(mStartTask->mBoxListYZBuffer);
	E_FREE(mStartTask->mBoxListXBuffer);
}

EHDI ABP_CompleteBoxPruningStartTask::ABP_CompleteBoxPruningStartTask() :
	mListX			(NULL),
	mListYZ			(NULL),
	mInputRemap		(NULL),
	mPairManager	(NULL),
	mRemap			(NULL),
	mBoxListXBuffer	(NULL),
	mBoxListYZBuffer(NULL),
	mNb				(0)
{
}

EHDI void ABP_CompleteBoxPruningStartTask::setup(
	//ABP_MM& memoryManager,
	const PxBounds3& updatedBounds,
	ABP_PairManager*  pairManager,
	PxU32 nb,
	const SIMD_AABB_X4*  listX,
	const SIMD_AABB_YZ4*  listYZ,
	const ABP_Index*  inputRemap,
	PxU64 contextID)
{
	mListX			= listX;
	mListYZ			= listYZ;
	mInputRemap		= inputRemap;
	mPairManager	= pairManager;

	mBounds = updatedBounds;
	mContextID = contextID;
	mNb = nb;

	// PT: TODO: revisit allocs
	//mBoxListXBuffer = reinterpret_cast<SIMD_AABB_X4*>(memoryManager.frameAlloc(sizeof(SIMD_AABB_X4)*(nb+NB_SENTINELS*NB_BUCKETS)));
	//mBoxListYZBuffer = reinterpret_cast<SIMD_AABB_YZ4*>(memoryManager.frameAlloc(sizeof(SIMD_AABB_YZ4)*nb));
	mBoxListXBuffer = reinterpret_cast<SIMD_AABB_X4*>(EPX_ALLOC(sizeof(SIMD_AABB_X4)*(nb+NB_SENTINELS*NB_BUCKETS), "mBoxListXBuffer"));
	mBoxListYZBuffer = reinterpret_cast<SIMD_AABB_YZ4*>(EPX_ALLOC(sizeof(SIMD_AABB_YZ4)*nb, "mBoxListYZBuffer"));

	//mRemap = reinterpret_cast<PxU32*>(memoryManager.frameAlloc(sizeof(PxU32)*nb));
	mRemap = reinterpret_cast<PxU32*>(EPX_ALLOC(sizeof(PxU32)*nb, "mRemap"));

	mEndTask.mStartTask = this;
	for(PxU32 i=0;i<9;i++)
		mTasks[i].mStartTask = this;
}

EHDI void ABP_CompleteBoxPruningStartTask::run()
{
//	printf("Running ABP_CompleteBoxPruningStartTask\n");

	const SIMD_AABB_X4*  listX = mListX;
	const SIMD_AABB_YZ4*  listYZ = mListYZ;
	const ABP_Index*  remap = mInputRemap;

	const PxU32 nb = mNb;
	PxU32*  Remap = mRemap;
	SIMD_AABB_X4*  BoxListXBuffer = mBoxListXBuffer;
	SIMD_AABB_YZ4*  BoxListYZBuffer = mBoxListYZBuffer;
	PxU32*  Counters = mCounters;

	SIMD_AABB_X4**  BoxListX = mBoxListX;
	SIMD_AABB_YZ4**  BoxListYZ = mBoxListYZ;
	PxU32**  RemapBase = mRemapBase;
	{
		EPX_PROFILE_ZONE("ABP_CompleteBoxPruningStartTask - Run", mContextID);

		// PT: TODO: revisit allocs
		//BoxListXBuffer = reinterpret_cast<SIMD_AABB_X4*>(memoryManager.frameAlloc(sizeof(SIMD_AABB_X4)*(nb+NB_SENTINELS*NB_BUCKETS)));
		//BoxListYZBuffer = reinterpret_cast<SIMD_AABB_YZ4*>(memoryManager.frameAlloc(sizeof(SIMD_AABB_YZ4)*nb));

		const PxVec3& mergedMin = mBounds.minimum;
		const PxVec3& mergedMax = mBounds.maximum;
		const float limitY = (mergedMax[1] + mergedMin[1]) * 0.5f;
		const float limitZ = (mergedMax[2] + mergedMin[2]) * 0.5f;

		for(PxU32 i=0;i<NB_BUCKETS;i++)
			Counters[i] = 0;

		//Remap = reinterpret_cast<PxU32*>(memoryManager.frameAlloc(sizeof(PxU32)*nb));
		// PT: TODO: revisit allocs
		//PxU8* Indices = reinterpret_cast<PxU8*>(memoryManager.frameAlloc(sizeof(PxU8)*nb));
		PxU8* Indices = reinterpret_cast<PxU8*>(EPX_ALLOC(sizeof(PxU8)*nb, "Indices"));
		{
			EPX_PROFILE_ZONE("BoxPruning - ClassifyBoxes", mContextID);
			for(PxU32 i=0;i<nb;i++)
			{
				const PxU8 index = classifyBoxNew(listYZ[i], limitY, limitZ);
				Indices[i] = index;
				Counters[index]++;
			}
		}

		{
			SIMD_AABB_X4* CurrentBoxListXBuffer = BoxListXBuffer;
			SIMD_AABB_YZ4* CurrentBoxListYZBuffer = BoxListYZBuffer;
			PxU32* CurrentRemap = Remap;
			for(PxU32 i=0;i<NB_BUCKETS;i++)
			{
				const PxU32 Nb = Counters[i];
				BoxListX[i] = CurrentBoxListXBuffer;
				BoxListYZ[i] = CurrentBoxListYZBuffer;
				RemapBase[i] = CurrentRemap;
				CurrentBoxListXBuffer += Nb+NB_SENTINELS;
				CurrentBoxListYZBuffer += Nb;
				CurrentRemap += Nb;
			}
			E_ASSERT(CurrentBoxListXBuffer == BoxListXBuffer + nb + NB_SENTINELS*NB_BUCKETS);
			E_ASSERT(CurrentBoxListYZBuffer == BoxListYZBuffer + nb);
			E_ASSERT(CurrentRemap == Remap + nb);
		}

		for(PxU32 i=0;i<NB_BUCKETS;i++)
			Counters[i] = 0;

		for(PxU32 i=0;i<nb;i++)
		{
			const PxU32 SortedIndex = i;
			const PxU32 TargetBucket = PxU32(Indices[SortedIndex]);

			const PxU32 IndexInTarget = Counters[TargetBucket]++;

			SIMD_AABB_X4* TargetBoxListX = BoxListX[TargetBucket];
			SIMD_AABB_YZ4* TargetBoxListYZ = BoxListYZ[TargetBucket];
			PxU32* TargetRemap = RemapBase[TargetBucket];

			TargetRemap[IndexInTarget] = remap[SortedIndex];
			TargetBoxListX[IndexInTarget] = listX[SortedIndex];
			TargetBoxListYZ[IndexInTarget] = listYZ[SortedIndex];
		}
		//memoryManager.frameFree(Indices);
		E_FREE(Indices);

		for(PxU32 i=0;i<NB_BUCKETS;i++)
		{
			SIMD_AABB_X4* TargetBoxListX = BoxListX[i];
			const PxU32 IndexInTarget = Counters[i];
			for(PxU32 j=0;j<NB_SENTINELS;j++)
				TargetBoxListX[IndexInTarget+j].initSentinel();
		}
	}

	for(PxU32 i=0;i<8;i++)
	{
		mTasks[i].mCounter = Counters[i/2];
		mTasks[i].mBoxListX = BoxListX[i/2];
		mTasks[i].mBoxListYZ = BoxListYZ[i/2];
		mTasks[i].mRemap = RemapBase[i/2];
		mTasks[i].mType = i&1;

		mTasks[i].mCounter4 = Counters[4];
		mTasks[i].mBoxListX4 = BoxListX[4];
		mTasks[i].mBoxListYZ4 = BoxListYZ[4];
		mTasks[i].mRemap4 = RemapBase[4];

		mTasks[i].mPairs.mSharedPM = mPairManager;
		//mTasks[i].mPairs.mDelayedPairs.reserve(10000);
	}

	PxU32 i=8;
	{
		mTasks[i].mCounter = Counters[4];
		mTasks[i].mBoxListX = BoxListX[4];
		mTasks[i].mBoxListYZ = BoxListYZ[4];
		mTasks[i].mRemap = RemapBase[4];
		mTasks[i].mType = 0;

		mTasks[i].mCounter4 = Counters[4];
		mTasks[i].mBoxListX4 = BoxListX[4];
		mTasks[i].mBoxListYZ4 = BoxListYZ[4];
		mTasks[i].mRemap4 = RemapBase[4];

		mTasks[i].mPairs.mSharedPM = mPairManager;
		//mTasks[i].mPairs.mDelayedPairs.reserve(10000);
	}

	for(PxU32 k=0; k<8+1; k++)
	{
		if(mTasks[k].isThereWorkToDo())
		{
			mTasks[k].mID = PxU16(k);
			mTasks[k].setContinuation(getContinuation());
		}
	}

	for(PxU32 k=0; k<8+1; k++)
	{
		if(mTasks[k].isThereWorkToDo())
			mTasks[k].removeReference();
	}
}

EHDI void ABP_CompleteBoxPruningStartTask::addDelayedPairs()
{
	EPX_PROFILE_ZONE("ABP_CompleteBoxPruningStartTask - add delayed pairs", mContextID);

	PxU32 nbDelayedPairs = 0;
	for(PxU32 k=0; k<9; k++)
		nbDelayedPairs += mTasks[k].mPairs.mDelayedPairs.size();

	if(nbDelayedPairs)
	{
		{
			EPX_PROFILE_ZONE("BroadPhaseABP - resizeForNewPairs", mContextID);
			mPairManager->resizeForNewPairs(nbDelayedPairs);
		}

		for(PxU32 k=0; k<9; k++)
			mPairManager->addDelayedPairs(mTasks[k].mPairs.mDelayedPairs);
	}
}

EHDI void ABP_CompleteBoxPruningStartTask::addDelayedPairs2(PxArray<BroadPhasePair>& createdPairs)
{
	EPX_PROFILE_ZONE("ABP_CompleteBoxPruningStartTask - add delayed pairs", mContextID);

	PxU32 nbDelayedPairs = 0;
	for(PxU32 k=0; k<9; k++)
		nbDelayedPairs += mTasks[k].mPairs.mDelayedPairs.size();

	if(nbDelayedPairs)
	{
		{
			EPX_PROFILE_ZONE("BroadPhaseABP - resizeForNewPairs", mContextID);
			mPairManager->resizeForNewPairs(nbDelayedPairs);
		}

		for(PxU32 k=0; k<9; k++)
			mPairManager->addDelayedPairs2(createdPairs, mTasks[k].mPairs.mDelayedPairs);
	}
}

EHDI static void CompleteBoxPruning_Version16(
	ABP_CompleteBoxPruningStartTask& completeBoxPruningTask,
	ABP_MM& memoryManager,
	const PxBounds3& updatedBounds,
	ABP_PairManager*  pairManager,
	PxU32 nb,
	const SIMD_AABB_X4*  listX,
	const SIMD_AABB_YZ4*  listYZ,
	const ABP_Index*  remap,
	PxBaseTask* continuation, PxU64 contextID)
{
	E_UNUSED(contextID);
	E_UNUSED(continuation);

	if(!nb)
		return;

	if(continuation)
	{
		completeBoxPruningTask.setup(updatedBounds, pairManager, nb, listX, listYZ, remap, contextID);

		completeBoxPruningTask.mEndTask.setContinuation(continuation);
		completeBoxPruningTask.setContinuation(&completeBoxPruningTask.mEndTask);

		completeBoxPruningTask.mEndTask.removeReference();
		completeBoxPruningTask.removeReference();
		return;
	}

	PxU32* Remap;
	SIMD_AABB_X4* BoxListXBuffer;
	SIMD_AABB_YZ4* BoxListYZBuffer;
	PxU32 Counters[NB_BUCKETS];
	SIMD_AABB_X4* BoxListX[NB_BUCKETS];
	SIMD_AABB_YZ4* BoxListYZ[NB_BUCKETS];
	PxU32* RemapBase[NB_BUCKETS];
	{
		EPX_PROFILE_ZONE("BoxPruning - PrepareData", contextID);

		// PT: TODO: revisit allocs
		BoxListXBuffer = reinterpret_cast<SIMD_AABB_X4*>(memoryManager.frameAlloc(sizeof(SIMD_AABB_X4)*(nb+NB_SENTINELS*NB_BUCKETS)));
		BoxListYZBuffer = reinterpret_cast<SIMD_AABB_YZ4*>(memoryManager.frameAlloc(sizeof(SIMD_AABB_YZ4)*nb));

		const PxVec3& mergedMin = updatedBounds.minimum;
		const PxVec3& mergedMax = updatedBounds.maximum;
		const float limitY = (mergedMax[1] + mergedMin[1]) * 0.5f;
		const float limitZ = (mergedMax[2] + mergedMin[2]) * 0.5f;

		for(PxU32 i=0;i<NB_BUCKETS;i++)
			Counters[i] = 0;

		Remap = reinterpret_cast<PxU32*>(memoryManager.frameAlloc(sizeof(PxU32)*nb));
		PxU8* Indices = reinterpret_cast<PxU8*>(memoryManager.frameAlloc(sizeof(PxU8)*nb));
		{
			EPX_PROFILE_ZONE("BoxPruning - ClassifyBoxes", contextID);
			for(PxU32 i=0;i<nb;i++)
			{
				const PxU8 index = classifyBoxNew(listYZ[i], limitY, limitZ);
				Indices[i] = index;
				Counters[index]++;
			}
		}

		{
			SIMD_AABB_X4* CurrentBoxListXBuffer = BoxListXBuffer;
			SIMD_AABB_YZ4* CurrentBoxListYZBuffer = BoxListYZBuffer;
			PxU32* CurrentRemap = Remap;
			for(PxU32 i=0;i<NB_BUCKETS;i++)
			{
				const PxU32 Nb = Counters[i];
				BoxListX[i] = CurrentBoxListXBuffer;
				BoxListYZ[i] = CurrentBoxListYZBuffer;
				RemapBase[i] = CurrentRemap;
				CurrentBoxListXBuffer += Nb+NB_SENTINELS;
				CurrentBoxListYZBuffer += Nb;
				CurrentRemap += Nb;
			}
			E_ASSERT(CurrentBoxListXBuffer == BoxListXBuffer + nb + NB_SENTINELS*NB_BUCKETS);
			E_ASSERT(CurrentBoxListYZBuffer == BoxListYZBuffer + nb);
			E_ASSERT(CurrentRemap == Remap + nb);
		}

		for(PxU32 i=0;i<NB_BUCKETS;i++)
			Counters[i] = 0;

		for(PxU32 i=0;i<nb;i++)
		{
			const PxU32 SortedIndex = i;
			const PxU32 TargetBucket = PxU32(Indices[SortedIndex]);

			const PxU32 IndexInTarget = Counters[TargetBucket]++;

			SIMD_AABB_X4* TargetBoxListX = BoxListX[TargetBucket];
			SIMD_AABB_YZ4* TargetBoxListYZ = BoxListYZ[TargetBucket];
			PxU32* TargetRemap = RemapBase[TargetBucket];

			TargetRemap[IndexInTarget] = remap[SortedIndex];
			TargetBoxListX[IndexInTarget] = listX[SortedIndex];
			TargetBoxListYZ[IndexInTarget] = listYZ[SortedIndex];
		}
		memoryManager.frameFree(Indices);

		for(PxU32 i=0;i<NB_BUCKETS;i++)
		{
			SIMD_AABB_X4* TargetBoxListX = BoxListX[i];
			const PxU32 IndexInTarget = Counters[i];
			for(PxU32 j=0;j<NB_SENTINELS;j++)
				TargetBoxListX[IndexInTarget+j].initSentinel();
		}
	}

	{
		for(PxU32 i=0;i<NB_BUCKETS;i++)
		{
				doCompleteBoxPruning_Leaf(	pairManager,
										Counters[i],
										BoxListX[i], BoxListYZ[i],
										RemapBase[i]);
		}

		for(PxU32 i=0;i<NB_BUCKETS-1;i++)
		{
			doBipartiteBoxPruning_Leaf(pairManager,
				Counters[i], Counters[NB_BUCKETS-1],
				BoxListX[i], BoxListX[NB_BUCKETS-1], BoxListYZ[i], BoxListYZ[NB_BUCKETS-1],
				RemapBase[i], RemapBase[NB_BUCKETS-1]
				);
		}
	}

	memoryManager.frameFree(Remap);
	memoryManager.frameFree(BoxListYZBuffer);
	memoryManager.frameFree(BoxListXBuffer);
}


EHDI static void doCompleteBoxPruning_(
	ABP_CompleteBoxPruningStartTask& completeBoxPruningTask,
	ABP_CompleteBoxPruningTask&	bipTask0,
	ABP_CompleteBoxPruningTask&	bipTask1,
	ABP_MM& memoryManager, ABP_PairManager*  pairManager, const DynamicManager& mDBM, PxBaseTask* continuation, PxU64 contextID)
{
	const PxU32 nbUpdated = mDBM.getNbUpdatedBoxes();
	if(!nbUpdated)
		return;
	const PxU32 nbNonUpdated = mDBM.getNbNonUpdatedBoxes();
	const DynamicBoxes& updatedBoxes = mDBM.getUpdatedBoxes();

	const SIMD_AABB_X4*  updatedDynamicBoxes_X = updatedBoxes.getBoxes_X();
	const SIMD_AABB_YZ4*  updatedDynamicBoxes_YZ = updatedBoxes.getBoxes_YZ();

	// PT: find sleeping-dynamics-vs-active-dynamics overlaps
	if(nbNonUpdated)
	{
		if(continuation)
		{
			bipTask0.mCounter = nbUpdated;
			bipTask0.mBoxListX = updatedBoxes.getBoxes_X();
			bipTask0.mBoxListYZ = updatedBoxes.getBoxes_YZ();
			bipTask0.mRemap = mDBM.getRemap_Updated();
			bipTask0.mType = 1;

			bipTask0.mCounter4 = nbNonUpdated;
			bipTask0.mBoxListX4 = mDBM.getSleepingBoxes().getBoxes_X();
			bipTask0.mBoxListYZ4 = mDBM.getSleepingBoxes().getBoxes_YZ();
			bipTask0.mRemap4 = mDBM.getRemap_Sleeping();

			bipTask0.mPairs.mSharedPM = pairManager;
			//bipTask0.mPairs.mDelayedPairs.reserve(10000);

			if(bipTask0.isThereWorkToDo())
			{
				bipTask0.mID = 0;
				bipTask0.setContinuation(continuation);
				bipTask0.removeReference();
			}
		}
		else
		doBipartiteBoxPruning_Leaf(	pairManager, nbUpdated, nbNonUpdated,
									updatedBoxes, mDBM.getSleepingBoxes(),
									mDBM.getRemap_Updated(), mDBM.getRemap_Sleeping());
	}

	///////

	// PT: find active-dynamics-vs-active-dynamics overlaps
	if(1)
	{
		E_UNUSED(memoryManager);
		if(nbUpdated>USE_ABP_BUCKETS)
			CompleteBoxPruning_Version16(
											completeBoxPruningTask,
											memoryManager, mDBM.getUpdatedBounds(), pairManager, nbUpdated,
											updatedDynamicBoxes_X, updatedDynamicBoxes_YZ,
											mDBM.getRemap_Updated(), continuation, contextID);
		else
		{
			if(continuation)
			{
				bipTask1.mCounter = nbUpdated;
				bipTask1.mBoxListX = updatedDynamicBoxes_X;
				bipTask1.mBoxListYZ = updatedDynamicBoxes_YZ;
				bipTask1.mRemap = mDBM.getRemap_Updated();
				bipTask1.mType = 0;

				bipTask1.mPairs.mSharedPM = pairManager;
				//bipTask1.mPairs.mDelayedPairs.reserve(10000);

				if(bipTask1.isThereWorkToDo())
				{
					bipTask1.mID = 0;
					bipTask1.setContinuation(continuation);
					bipTask1.removeReference();
				}
			}
			else
			doCompleteBoxPruning_Leaf(	pairManager, nbUpdated,
										updatedDynamicBoxes_X, updatedDynamicBoxes_YZ,
										mDBM.getRemap_Updated());
		}
	}
}

EHDI void ABP::Region_prepareOverlaps()
{
	EPX_PROFILE_ZONE("ABP - Region_prepareOverlaps", mContextID);

	if(		!mDBM.isThereWorkToDo()
		&&	!mKBM.isThereWorkToDo()
		&&	!mSBM.isThereWorkToDo()
		)
		return;

	if(mSBM.isThereWorkToDo())
		mSBM.prepareData(mRS, mShared.mABP_Objects, mShared.mABP_Objects_Capacity, mMM, mContextID);

	mDBM.prepareData(mRS, mShared.mABP_Objects, mShared.mABP_Objects_Capacity, mMM, mContextID);
	mKBM.prepareData(mRS, mShared.mABP_Objects, mShared.mABP_Objects_Capacity, mMM, mContextID);

	mRS.reset();
}

// Finds static-vs-dynamic and dynamic-vs-dynamic overlaps
EHDI static void findAllOverlaps(
	ABP_CompleteBoxPruningStartTask& completeBoxPruningTask,
	ABP_CompleteBoxPruningTask&	bipTask0,
	ABP_CompleteBoxPruningTask&	bipTask1,
	ABP_CompleteBoxPruningTask&	bipTask2,
	ABP_CompleteBoxPruningTask&	bipTask3,
	ABP_CompleteBoxPruningTask&	bipTask4,
	ABP_MM& memoryManager, ABP_PairManager& pairManager, const StaticManager& mSBM, const DynamicManager& mDBM, bool doComplete, bool doBipartite, PxBaseTask* continuation, PxU64 contextID)
{
	const PxU32 nbUpdatedBoxesDynamic = mDBM.getNbUpdatedBoxes();

	// PT: find dynamics-vs-dynamics overlaps
	if(doComplete)
		doCompleteBoxPruning_(
			completeBoxPruningTask,
			bipTask3,
			bipTask4,
			memoryManager, &pairManager, mDBM, continuation, contextID);

	// PT: find dynamics-vs-statics overlaps
	if(doBipartite)
	{
		const PxU32 nbUpdatedBoxesStatic = mSBM.getNbUpdatedBoxes();
		const PxU32 nbNonUpdatedBoxesStatic = mSBM.getNbNonUpdatedBoxes();
		const PxU32 nbNonUpdatedBoxesDynamic = mDBM.getNbNonUpdatedBoxes();

		// PT: in previous versions we did active-dynamics-vs-all-statics here.
		if(nbUpdatedBoxesDynamic)
		{
			if(nbUpdatedBoxesStatic)
			{
				// PT: active static vs active dynamic
				if(continuation)
				{
					bipTask0.mCounter = nbUpdatedBoxesDynamic;
					bipTask0.mBoxListX = mDBM.getUpdatedBoxes().getBoxes_X();
					bipTask0.mBoxListYZ = mDBM.getUpdatedBoxes().getBoxes_YZ();
					bipTask0.mRemap = mDBM.getRemap_Updated();
					bipTask0.mType = 1;

					bipTask0.mCounter4 = nbUpdatedBoxesStatic;
					bipTask0.mBoxListX4 = mSBM.getUpdatedBoxes().getBoxes_X();
					bipTask0.mBoxListYZ4 = mSBM.getUpdatedBoxes().getBoxes_YZ();
					bipTask0.mRemap4 = mSBM.getRemap_Updated();

					bipTask0.mPairs.mSharedPM = &pairManager;
					//bipTask0.mPairs.mDelayedPairs.reserve(10000);

					if(bipTask0.isThereWorkToDo())
					{
						bipTask0.mID = 0;
						bipTask0.setContinuation(continuation);
						bipTask0.removeReference();
					}
				}
				else
				doBipartiteBoxPruning_Leaf(	&pairManager,
											nbUpdatedBoxesDynamic, nbUpdatedBoxesStatic,
											mDBM.getUpdatedBoxes(), mSBM.getUpdatedBoxes(),
											mDBM.getRemap_Updated(), mSBM.getRemap_Updated());
			}

			if(nbNonUpdatedBoxesStatic)
			{
				// PT: sleeping static vs active dynamic
				if(continuation)
				{
					bipTask1.mCounter = nbUpdatedBoxesDynamic;
					bipTask1.mBoxListX = mDBM.getUpdatedBoxes().getBoxes_X();
					bipTask1.mBoxListYZ = mDBM.getUpdatedBoxes().getBoxes_YZ();
					bipTask1.mRemap = mDBM.getRemap_Updated();
					bipTask1.mType = 1;

					bipTask1.mCounter4 = nbNonUpdatedBoxesStatic;
					bipTask1.mBoxListX4 = mSBM.getSleepingBoxes().getBoxes_X();
					bipTask1.mBoxListYZ4 = mSBM.getSleepingBoxes().getBoxes_YZ();
					bipTask1.mRemap4 = mSBM.getRemap_Sleeping();

					bipTask1.mPairs.mSharedPM = &pairManager;
					//bipTask1.mPairs.mDelayedPairs.reserve(10000);

					if(bipTask1.isThereWorkToDo())
					{
						bipTask1.mID = 0;
						bipTask1.setContinuation(continuation);
						bipTask1.removeReference();
					}
				}
				else
				doBipartiteBoxPruning_Leaf(	&pairManager,
											nbUpdatedBoxesDynamic, nbNonUpdatedBoxesStatic,
											mDBM.getUpdatedBoxes(), mSBM.getSleepingBoxes(),
											mDBM.getRemap_Updated(), mSBM.getRemap_Sleeping());
			}
		}

		if(nbUpdatedBoxesStatic && nbNonUpdatedBoxesDynamic)
		{
			// PT: active static vs sleeping dynamic
			if(continuation)
			{
				bipTask2.mCounter = nbNonUpdatedBoxesDynamic;
				bipTask2.mBoxListX = mDBM.getSleepingBoxes().getBoxes_X();
				bipTask2.mBoxListYZ = mDBM.getSleepingBoxes().getBoxes_YZ();
				bipTask2.mRemap = mDBM.getRemap_Sleeping();
				bipTask2.mType = 1;

				bipTask2.mCounter4 = nbUpdatedBoxesStatic;
				bipTask2.mBoxListX4 = mSBM.getUpdatedBoxes().getBoxes_X();
				bipTask2.mBoxListYZ4 = mSBM.getUpdatedBoxes().getBoxes_YZ();
				bipTask2.mRemap4 = mSBM.getRemap_Updated();

				bipTask2.mPairs.mSharedPM = &pairManager;
				//bipTask2.mPairs.mDelayedPairs.reserve(10000);

				if(bipTask2.isThereWorkToDo())
				{
					bipTask2.mID = 0;
					bipTask2.setContinuation(continuation);
					bipTask2.removeReference();
				}
			}
			else
			doBipartiteBoxPruning_Leaf(	&pairManager,
										nbNonUpdatedBoxesDynamic, nbUpdatedBoxesStatic,
										mDBM.getSleepingBoxes(), mSBM.getUpdatedBoxes(),
										mDBM.getRemap_Sleeping(), mSBM.getRemap_Updated());
		}
	}
}

///////////////////////////////////////////////////////////////////////////

EHDI ABP::ABP(PxU64 contextID) :
	mSBM		(FilterType::STATIC),
	mDBM		(FilterType::DYNAMIC),
	mKBM		(FilterType::KINEMATIC),
	mContextID	(contextID)
	,mTask0		(ABP_TASK_0)
	,mTask1		(ABP_TASK_1)
{
	mTask0.setContextId(mContextID);
	mTask1.setContextId(mContextID);
	mCompleteBoxPruningTask0.setContextId(mContextID);
	mCompleteBoxPruningTask1.setContextId(mContextID);
	for(PxU32 k=0; k<9; k++)
	{
		mCompleteBoxPruningTask0.mTasks[k].setContextId(mContextID);
		mCompleteBoxPruningTask1.mTasks[k].setContextId(mContextID);
	}

	for(PxU32 k=0; k<NB_BIP_TASKS; k++)
		mBipTasks[k].setContextId(mContextID);
}

EHDI ABP::~ABP()
{
	reset();
}

EHDI void ABP::freeBuffers()
{
	mShared.mRemovedObjects.empty();
}

EHDI void ABP::preallocate(PxU32 nbObjects, PxU32 maxNbOverlaps)
{
	if(nbObjects)
	{
		EPX_DELETE_ARRAY(mShared.mABP_Objects);
		ABP_Object* objects = EPX_NEW(ABP_Object)[nbObjects];
		mShared.mABP_Objects = objects;
		mShared.mABP_Objects_Capacity = nbObjects;
	}

	// PT: TODO: here we should preallocate the box arrays but we don't know how many of them will be static / dynamic...

	mPairManager.reserveMemory(maxNbOverlaps);
}

EHDI void ABP::addStaticObjects(const BpHandle* userID_, PxU32 nb, PxU32 maxID)
{
	mShared.checkResize(maxID);

	mSBM.addObjects(userID_, nb, NULL);
}

EHDI void ABP::addDynamicObjects(const BpHandle* userID_, PxU32 nb, PxU32 maxID)
{
	mShared.checkResize(maxID);

	mShared.mUpdatedObjects.checkResize(maxID);

	mDBM.addObjects(userID_, nb, &mShared);
}

EHDI void ABP::addKinematicObjects(const BpHandle* userID_, PxU32 nb, PxU32 maxID)
{
	mShared.checkResize(maxID);

	mShared.mUpdatedObjects.checkResize(maxID);

	mKBM.addObjects(userID_, nb, &mShared);
}

EHDI void ABP::removeObject(BpHandle userID)
{
	mShared.mUpdatedObjects.setBitChecked(userID);
	mShared.mRemovedObjects.setBitChecked(userID);

	E_ASSERT(userID<mShared.mABP_Objects_Capacity);
	ABPEntry& object = mShared.mABP_Objects[userID];

	// PT: TODO better
	BoxManager* bm;
	const FilterType::Enum objectType = object.getType();
	if(objectType==FilterType::STATIC)
	{
		bm = &mSBM;
	}
	else if(objectType==FilterType::KINEMATIC)
	{
		bm = &mKBM;
	}
	else
	{
		bm = &mDBM;
	}
	bm->removeObject(object, userID);

	object.invalidateIndex();
}

EHDI void ABP::updateObject(BpHandle userID)
{
	mShared.mUpdatedObjects.setBitChecked(userID);

	E_ASSERT(userID<mShared.mABP_Objects_Capacity);
	ABPEntry& object = mShared.mABP_Objects[userID];

	// PT: TODO better
	BoxManager* bm;
	const FilterType::Enum objectType = object.getType();
	if(objectType==FilterType::STATIC)
	{
		bm = &mSBM;
	}
	else if(objectType==FilterType::KINEMATIC)
	{
		bm = &mKBM;
	}
	else
	{
		bm = &mDBM;
	}
	bm->updateObject(object, userID);
}

// PT: TODO: replace bits with timestamps?
EHDI void ABP_PairManager::computeCreatedDeletedPairs(PxArray<BroadPhasePair>& createdPairs, PxArray<BroadPhasePair>& deletedPairs, const BitArray& updated, const BitArray& removed)
{
	// PT: parse all currently active pairs. The goal here is to generate the found/lost pairs, compared to previous frame.
	// PT: TODO: MT?
	PxU32 i=0;
	PxU32 nbActivePairs = mNbActivePairs;
	while(i<nbActivePairs)
	{
		InternalPair& p = mActivePairs[i];

		if(p.isNew())
		{
			// New pair

			// PT: 'isNew' is set to true in the 'addPair' function. In this case the pair did not previously
			// exist in the structure, and thus we must report the new pair to the client code.
			//
			// PT: group-based filtering is not needed here, since it has already been done in 'addPair'
			const PxU32 id0 = p.getId0();
			const PxU32 id1 = p.getId1();
			E_ASSERT(id0!=INVALID_ID);
			E_ASSERT(id1!=INVALID_ID);
			//createdPairs.pushBack(BroadPhasePair(id0, id1));
			BroadPhasePair* newPair = Cm::reserveContainerMemory(createdPairs, 1);
			newPair->mVolA = id0;
			newPair->mVolB = id1;
			// PT: TODO: replace this with bitmaps?
			p.clearNew();
			p.clearUpdated();
			i++;
		}
		else if(p.isUpdated())
		{
			// Persistent pair

			// PT: this pair already existed in the structure, and has been found again this frame. Since
			// MBP reports "all pairs" each frame (as opposed to SAP), this happens quite often, for each
			// active persistent pair.
			p.clearUpdated();
			i++;
		}
		else
		{
			// Lost pair

			// PT: if the pair is not new and not 'updated', it might be a lost (separated) pair. But this
			// is not always the case since we now handle "sleeping" objects directly within MBP. A pair
			// of sleeping objects does not generate an 'addPair' call, so it ends up in this codepath.
			// Nonetheless the sleeping pair should not be deleted. We can only delete pairs involving
			// objects that have been actually moved during the frame. This is the only case in which
			// a pair can indeed become 'lost'.
			const PxU32 id0 = p.getId0();
			const PxU32 id1 = p.getId1();
			E_ASSERT(id0!=INVALID_ID);
			E_ASSERT(id1!=INVALID_ID);

			// PT: if none of the involved objects have been updated, the pair is just sleeping: keep it and skip it.
			if(updated.isSetChecked(id0) || updated.isSetChecked(id1))
			{
				// PT: by design (for better or worse) we do not report pairs to the client when
				// one of the involved objects has been deleted. The pair must still be deleted
				// from the MBP structure though.
				if(!removed.isSetChecked(id0) && !removed.isSetChecked(id1))
				{
					// PT: doing the group-based filtering here is useless. The pair should not have
					// been added in the first place.
					//deletedPairs.pushBack(BroadPhasePair(id0, id1));
					BroadPhasePair* lostPair = Cm::reserveContainerMemory(deletedPairs, 1);
					lostPair->mVolA = id0;
					lostPair->mVolB = id1;
				}

				const PxU32 hashValue = hash(id0, id1) & mMask;
				removePair(id0, id1, hashValue, i);
				nbActivePairs--;
			}
			else i++;
		}
	}

	shrinkMemory();
}

EHDI void ABP::findOverlaps(PxBaseTask* continuation, const Bp::FilterGroup::Enum*  groups, const bool*  lut)
{
	EPX_PROFILE_ZONE("ABP - findOverlaps", mContextID);

	mPairManager.mGroups = groups;
	mPairManager.mLUT = lut;

	if(!gPrepareOverlapsFlag)
		Region_prepareOverlaps();

	bool doKineKine = true;
	bool doStaticKine = true;
	{
		doStaticKine = lut[Bp::FilterType::KINEMATIC*Bp::FilterType::COUNT + Bp::FilterType::STATIC];
		doKineKine = lut[Bp::FilterType::KINEMATIC*Bp::FilterType::COUNT + Bp::FilterType::KINEMATIC];
	}

	// Static-vs-dynamic (bipartite) and dynamic-vs-dynamic (complete)
	findAllOverlaps(
		mCompleteBoxPruningTask0,
		mBipTasks[0],
		mBipTasks[1],
		mBipTasks[2],
		mBipTasks[3],
		mBipTasks[4],
		mMM, mPairManager, mSBM, mDBM, true, true, continuation, mContextID);

	// Static-vs-kinematics (bipartite) and kinematics-vs-kinematics (complete)
	findAllOverlaps(
		mCompleteBoxPruningTask1,
		mBipTasks[5],
		mBipTasks[6],
		mBipTasks[7],
		mBipTasks[8],
		mBipTasks[9],
		mMM, mPairManager, mSBM, mKBM, doKineKine, doStaticKine, continuation, mContextID);

	if(1)
	{
		findAllOverlaps(
			mCompleteBoxPruningTask1,
			mBipTasks[10],
			mBipTasks[11],
			mBipTasks[12],
			mBipTasks[13],
			mBipTasks[14],
			mMM, mPairManager, mKBM, mDBM, false, true, continuation, mContextID);
	}
	else
	{
		const PxU32 nbUpdatedDynamics = mDBM.getNbUpdatedBoxes();
		const PxU32 nbNonUpdatedDynamics = mDBM.getNbNonUpdatedBoxes();
		const PxU32 nbUpdatedKinematics = mKBM.getNbUpdatedBoxes();
		const PxU32 nbNonUpdatedKinematics = mKBM.getNbNonUpdatedBoxes();

		if(nbUpdatedDynamics)
		{
			// Active dynamics vs active kinematics
			if(nbUpdatedKinematics)
			{
				doBipartiteBoxPruning_Leaf(	&mPairManager,
											nbUpdatedDynamics, nbUpdatedKinematics,
											mDBM.getUpdatedBoxes(), mKBM.getUpdatedBoxes(),
											mDBM.getRemap_Updated(), mKBM.getRemap_Updated());
			}
			// Active dynamics vs inactive kinematics
			if(nbNonUpdatedKinematics)
			{
				doBipartiteBoxPruning_Leaf(	&mPairManager,
											nbUpdatedDynamics, nbNonUpdatedKinematics,
											mDBM.getUpdatedBoxes(), mKBM.getSleepingBoxes(),
											mDBM.getRemap_Updated(), mKBM.getRemap_Sleeping());
			}
		}

		if(nbUpdatedKinematics && nbNonUpdatedDynamics)
		{
			// Inactive dynamics vs active kinematics
			doBipartiteBoxPruning_Leaf(	&mPairManager,
										nbNonUpdatedDynamics, nbUpdatedKinematics,
										mDBM.getSleepingBoxes(), mKBM.getUpdatedBoxes(),
										mDBM.getRemap_Sleeping(), mKBM.getRemap_Updated());
		}
	}
}

EHDI PxU32 ABP::finalize(PxArray<BroadPhasePair>& createdPairs, PxArray<BroadPhasePair>& deletedPairs)
{
	EPX_PROFILE_ZONE("ABP - finalize", mContextID);

	{
		EPX_PROFILE_ZONE("computeCreatedDeletedPairs", mContextID);

		mPairManager.computeCreatedDeletedPairs(createdPairs, deletedPairs, mShared.mUpdatedObjects, mShared.mRemovedObjects);
	}

	mShared.mUpdatedObjects.clearAll();

	return mPairManager.mNbActivePairs;
}

EHDI void ABP::addDelayedPairs()
{
	EPX_PROFILE_ZONE("ABP - addDelayedPairs", mContextID);

	mCompleteBoxPruningTask0.addDelayedPairs();
	mCompleteBoxPruningTask1.addDelayedPairs();

	PxU32 nbDelayedPairs = 0;
	for(PxU32 k=0; k<NB_BIP_TASKS; k++)
		nbDelayedPairs += mBipTasks[k].mPairs.mDelayedPairs.size();

	if(nbDelayedPairs)
	{
		{
			EPX_PROFILE_ZONE("ABP - resizeForNewPairs", mContextID);
			mPairManager.resizeForNewPairs(nbDelayedPairs);
		}

		for(PxU32 k=0; k<NB_BIP_TASKS; k++)
			mPairManager.addDelayedPairs(mBipTasks[k].mPairs.mDelayedPairs);
	}
}

EHDI void ABP::addDelayedPairs2(PxArray<BroadPhasePair>& createdPairs)
{
	EPX_PROFILE_ZONE("ABP - addDelayedPairs", mContextID);

	mCompleteBoxPruningTask0.addDelayedPairs2(createdPairs);
	mCompleteBoxPruningTask1.addDelayedPairs2(createdPairs);

	PxU32 nbDelayedPairs = 0;
	for(PxU32 k=0; k<NB_BIP_TASKS; k++)
		nbDelayedPairs += mBipTasks[k].mPairs.mDelayedPairs.size();

	if(nbDelayedPairs)
	{
		{
			EPX_PROFILE_ZONE("ABP - resizeForNewPairs", mContextID);
			mPairManager.resizeForNewPairs(nbDelayedPairs);
		}

		for(PxU32 k=0; k<NB_BIP_TASKS; k++)
			mPairManager.addDelayedPairs2(createdPairs, mBipTasks[k].mPairs.mDelayedPairs);
	}
}

EHDI void ABP::reset()
{
	mSBM.reset();
	mDBM.reset();
	mKBM.reset();

	EPX_DELETE_ARRAY(mShared.mABP_Objects);
	mShared.mABP_Objects_Capacity = 0;
	mPairManager.purge();
	mShared.mUpdatedObjects.empty();
	mShared.mRemovedObjects.empty();
}

// PT: TODO: is is really ok to use "transient" data in this function?
EHDI void ABP::shiftOrigin(const PxVec3& shift, const PxBounds3* /*boundsArray*/, const PxReal* /*contactDistances*/)
{
	E_UNUSED(shift);	// PT: unused because the bounds were pre-shifted before calling this function

	// PT: the AABB manager marks all objects as updated when we shift so the stuff below may not be necessary
}

EHDI void ABP::setTransientData(const PxBounds3* bounds, const PxReal* contactDistance)
{
	mSBM.setSourceData(bounds, contactDistance);
	mDBM.setSourceData(bounds, contactDistance);
	mKBM.setSourceData(bounds, contactDistance);
}
///////////////////////////////////////////////////////////////////////////////

}

// Below is the PhysX wrapper = link between AABBManager and ABP

using namespace internalABP;

#define DEFAULT_CREATED_DELETED_PAIRS_CAPACITY 1024

EHDI BroadPhaseABP::BroadPhaseABP(	PxU32 maxNbBroadPhaseOverlaps,
								PxU32 maxNbStaticShapes,
								PxU32 maxNbDynamicShapes,
								PxU64 contextID,
								bool enableMT) :
	mNbAdded		(0),
	mNbUpdated		(0),
	mNbRemoved		(0),
	mCreatedHandles	(NULL),
	mUpdatedHandles	(NULL),
	mRemovedHandles	(NULL),
	mGroups			(NULL),
	mFilter			(NULL),
	mContextID		(contextID),
	mEnableMT		(enableMT)
{
	mABP = EPX_NEW(ABP)(contextID);

	const PxU32 nbObjects = maxNbStaticShapes + maxNbDynamicShapes;
	mABP->preallocate(nbObjects, maxNbBroadPhaseOverlaps);

	mCreated.reserve(DEFAULT_CREATED_DELETED_PAIRS_CAPACITY);
	mDeleted.reserve(DEFAULT_CREATED_DELETED_PAIRS_CAPACITY);
}

EHDI BroadPhaseABP::~BroadPhaseABP()
{
	E_DELETE(mABP);
}

EHDI void BroadPhaseABP::update(PxcScratchAllocator* scratchAllocator, const BroadPhaseUpdateData& updateData, PxBaseTask* continuation)
{
	EPX_PROFILE_ZONE("BroadPhaseABP - update", mContextID);
	EPX_CHECK_AND_RETURN(scratchAllocator, "BroadPhaseABP::update - scratchAllocator must be non-NULL \n");

	{
		EPX_PROFILE_ZONE("BroadPhaseABP - setup", mContextID);

		mABP->mMM.mScratchAllocator = scratchAllocator;
		mABP->setTransientData(updateData.getAABBs(), updateData.getContactDistance());

		const PxU32 newCapacity = updateData.getCapacity();
		mABP->mShared.checkResize(newCapacity);


		mGroups			= updateData.getGroups();
		mFilter			= &updateData.getFilter();
		mNbAdded		= updateData.getNumCreatedHandles();
		mNbUpdated		= updateData.getNumUpdatedHandles();
		mNbRemoved		= updateData.getNumRemovedHandles();
		mCreatedHandles	= updateData.getCreatedHandles();
		mUpdatedHandles	= updateData.getUpdatedHandles();
		mRemovedHandles	= updateData.getRemovedHandles();
	}

	// PT: run single-threaded if forced to do so
	if(!mEnableMT)
		continuation = NULL;

	if(continuation)
	{
		mABP->mTask1.mBP = this;
		mABP->mTask1.setContinuation(continuation);

		mABP->mTask0.mBP = this;
		mABP->mTask0.setContinuation(&mABP->mTask1);

		mABP->mTask1.removeReference();
		mABP->mTask0.removeReference();
	}
	else
	{
		{
			EPX_PROFILE_ZONE("BroadPhaseABP - setUpdateData", mContextID);

			removeObjects();
			addObjects();
			updateObjects();

			E_ASSERT(!mCreated.size());
			E_ASSERT(!mDeleted.size());

			if(gPrepareOverlapsFlag)
				mABP->Region_prepareOverlaps();
		}

		{
			EPX_PROFILE_ZONE("BroadPhaseABP - update", mContextID);
			mABP->findOverlaps(continuation, mGroups, mFilter->getLUT());
		}

		{
			EPX_PROFILE_ZONE("BroadPhaseABP - postUpdate", mContextID);
			mABP->finalize(mCreated, mDeleted);
		}
	}
}

EHDI void ABP_InternalTask::run()
{
	EPX_SIMD_GUARD

	internalABP::ABP* abp = mBP->mABP;

	if(mID==ABP_TASK_0)
	{
		{
			EPX_PROFILE_ZONE("ABP_InternalTask - setUpdateData", mContextID);

			mBP->removeObjects();
			mBP->addObjects();
			mBP->updateObjects();

			E_ASSERT(!mBP->mCreated.size());
			E_ASSERT(!mBP->mDeleted.size());

			if(gPrepareOverlapsFlag)
				abp->Region_prepareOverlaps();
		}

		{
			EPX_PROFILE_ZONE("ABP_InternalTask - update", mContextID);

			for(PxU32 k=0;k<9;k++)
			{
				abp->mCompleteBoxPruningTask0.mTasks[k].mPairs.mDelayedPairs.resetOrClear();
				abp->mCompleteBoxPruningTask1.mTasks[k].mPairs.mDelayedPairs.resetOrClear();
			}

			for(PxU32 k=0;k<NB_BIP_TASKS;k++)
				abp->mBipTasks[k].mPairs.mDelayedPairs.resetOrClear();

			abp->findOverlaps(getContinuation(), mBP->mGroups, mBP->mFilter->getLUT());
		}
	}
	else if(mID==ABP_TASK_1)
	{
		//abp->addDelayedPairs();
		//abp->finalize(mBP->mCreated, mBP->mDeleted);
		abp->finalize(mBP->mCreated, mBP->mDeleted);
		abp->addDelayedPairs2(mBP->mCreated);
	}
}

EHDI void BroadPhaseABP::removeObjects()
{
	EPX_PROFILE_ZONE("BroadPhaseABP - removeObjects", mContextID);

	PxU32 nbRemoved = mNbRemoved;
	const BpHandle* removed = mRemovedHandles;

	if(!nbRemoved || !removed)
		return;

	while(nbRemoved--)
	{
		const BpHandle index = *removed++;
		E_ASSERT(index+1<mABP->mShared.mABP_Objects_Capacity);	// PT: we allocated one more box on purpose
		mABP->removeObject(index);
	}
}

EHDI void BroadPhaseABP::updateObjects()
{
	EPX_PROFILE_ZONE("BroadPhaseABP - updateObjects", mContextID);

	PxU32 nbUpdated = mNbUpdated;
	const BpHandle* updated = mUpdatedHandles;

	if(!nbUpdated || !updated)
		return;

	while(nbUpdated--)
	{
		const BpHandle index = *updated++;
		E_ASSERT(index+1<mABP->mShared.mABP_Objects_Capacity);	// PT: we allocated one more box on purpose
		mABP->updateObject(index);
	}
}

EHDI void BroadPhaseABP::addObjects()
{
	EPX_PROFILE_ZONE("BroadPhaseABP - addObjects", mContextID);

	PxU32 nbAdded = mNbAdded;
	const BpHandle* created = mCreatedHandles;

	if(!nbAdded || !created)
		return;

	const Bp::FilterGroup::Enum*  groups = mGroups;

	struct Batch
	{
		EHD	Batch() : mNb(0), mMaxIndex(0)	{}

		PxU32		mNb;
		PxU32		mMaxIndex;
		BpHandle	mIndices[ABP_BATCHING];

		EHD void add(const BpHandle index, internalABP::ABP*  abp, FilterType::Enum type)
		{
			PxU32 nb = mNb;
			mMaxIndex = PxMax(mMaxIndex, index);
			mIndices[nb++] = index;
			if(nb==ABP_BATCHING)
			{
				mNb = 0;
				// PT: TODO: we could use a function ptr here
				if(type==FilterType::STATIC)
					abp->addStaticObjects(mIndices, ABP_BATCHING, mMaxIndex);
				else if(type==FilterType::KINEMATIC)
					abp->addKinematicObjects(mIndices, ABP_BATCHING, mMaxIndex);
				else
				{
					E_ASSERT(type==FilterType::DYNAMIC || type==FilterType::AGGREGATE);
					abp->addDynamicObjects(mIndices, ABP_BATCHING, mMaxIndex);
				}

				mMaxIndex = 0;
			}
			else
				mNb = nb;
		}
	};
	Batch statics;
	Batch dynamics;
	Batch kinematics;

	Batch* batches[FilterType::COUNT] = {NULL};
	batches[FilterType::STATIC] = &statics;
	batches[FilterType::DYNAMIC] = &dynamics;
	batches[FilterType::AGGREGATE] = &dynamics;
	batches[FilterType::KINEMATIC] = &kinematics;

	while(nbAdded--)
	{
		const BpHandle index = *created++;
		E_ASSERT(index+1<mABP->mShared.mABP_Objects_Capacity);	// PT: we allocated one more box on purpose
		FilterType::Enum type = FilterType::Enum(groups[index] & BP_FILTERING_TYPE_MASK);
		if(!batches[type])
			type = FilterType::DYNAMIC;
		batches[type]->add(index, mABP, type);
	}

	if(statics.mNb)
		mABP->addStaticObjects(statics.mIndices, statics.mNb, statics.mMaxIndex);
	if(kinematics.mNb)
		mABP->addKinematicObjects(kinematics.mIndices, kinematics.mNb, kinematics.mMaxIndex);
	if(dynamics.mNb)
		mABP->addDynamicObjects(dynamics.mIndices, dynamics.mNb, dynamics.mMaxIndex);
}

EHDI const BroadPhasePair* BroadPhaseABP::getCreatedPairs(PxU32& nbCreatedPairs) const
{
	nbCreatedPairs = mCreated.size();
	return mCreated.begin();
}

EHDI const BroadPhasePair* BroadPhaseABP::getDeletedPairs(PxU32& nbDeletedPairs) const
{
	nbDeletedPairs = mDeleted.size();
	return mDeleted.begin();
}

EHDI static void freeBuffer(PxArray<BroadPhasePair>& buffer)
{
	const PxU32 size = buffer.size();
	if(size>DEFAULT_CREATED_DELETED_PAIRS_CAPACITY)
	{
		buffer.reset();
		buffer.reserve(DEFAULT_CREATED_DELETED_PAIRS_CAPACITY);
	}
	else
	{
		buffer.clear();
	}
}

EHDI void BroadPhaseABP::freeBuffers()
{
	EPX_PROFILE_ZONE("BroadPhaseABP - freeBuffers", mContextID);

	mABP->freeBuffers();
	freeBuffer(mCreated);
	freeBuffer(mDeleted);
}


EHDI void BroadPhaseABP::shiftOrigin(const PxVec3& shift, const PxBounds3* boundsArray, const PxReal* contactDistances)
{
	mABP->shiftOrigin(shift, boundsArray, contactDistances);
}

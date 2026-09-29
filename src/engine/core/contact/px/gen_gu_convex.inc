// ===== 원본: physx/include/foundation/PxSIMDHelpers.h
namespace px
{

	//! A padded version of PxMat33, to safely load its data using SIMD
	class PxMat33Padded : public PxMat33
	{
	public:
		explicit EHD PxMat33Padded(const PxQuat& q)
		{
			using namespace aos;
			const QuatV qV = V4LoadU(&q.x);
			Vec3V column0V, column1V, column2V;
			QuatGetMat33V(qV, column0V, column1V, column2V);
			V4StoreU(column0V, &column0.x);
			V4StoreU(column1V, &column1.x);
			V4StoreU(column2V, &column2.x);
		}
		EHD ~PxMat33Padded()				{}
		EHD void operator=(const PxMat33& other)
		{
			column0 = other.column0;
			column1 = other.column1;
			column2 = other.column2;
		}
		PxU32	padding;
	};

namespace aos
{

	EHD void transformKernelVec4(	const FloatVArg wa, const Vec4VArg va, const Vec4VArg pa, 
												const FloatVArg wb, const Vec4VArg vb, const Vec4VArg pb, 
												FloatV& wo, Vec4V& vo, Vec4V& po)
	{
		wo = FSub(FMul(wa, wb), V4Dot3(va, vb));
		vo = V4ScaleAdd(va, wb, V4ScaleAdd(vb, wa, V4Cross(va, vb)));

		const Vec4V t1 = V4Scale(pb, FScaleAdd(wa, wa, FLoad(-0.5f)));
		const Vec4V t2 = V4ScaleAdd(V4Cross(va, pb), wa, t1);
		const Vec4V t3 = V4ScaleAdd(va, V4Dot3(va, pb), t2);

		po = V4ScaleAdd(t3, FLoad(2.0f), pa);
	}

	// PT: out = a * b
	template<const bool alignedInput, const bool alignedOutput>
	EHD void transformMultiply(PxTransform& out, const PxTransform& a, const PxTransform& b)
	{
		E_ASSERT(!alignedInput || (size_t(&a)&15) == 0);
		E_ASSERT(!alignedInput || (size_t(&b)&15) == 0);

		const Vec4V aPos = alignedInput ? V4LoadA(&a.p.x) : V4LoadU(&a.p.x);
		const Vec4V aRot = alignedInput ? V4LoadA(&a.q.x) : V4LoadU(&a.q.x);

		const Vec4V bPos = alignedInput ? V4LoadA(&b.p.x) : V4LoadU(&b.p.x);
		const Vec4V bRot = alignedInput ? V4LoadA(&b.q.x) : V4LoadU(&b.q.x);
	
		Vec4V v, p;
		FloatV w; 
		transformKernelVec4(V4GetW(aRot), aRot, aPos, V4GetW(bRot), bRot, bPos, w, v, p);

		if(alignedOutput)
		{
			E_ASSERT((size_t(&out)&15) == 0);
			V4StoreA(p, &out.p.x);
			V4StoreA(V4SetW(v,w), &out.q.x);
		}
		else
		{
			V4StoreU(p, &out.p.x);
			V4StoreU(V4SetW(v,w), &out.q.x);
		}
	}

	// PT: out = a * b
	EHD void transformMultiply(PxTransform32& out, const PxTransform32& a, const PxTransform32& b)
	{
		transformMultiply<true, true>(out, a, b);
	}

} // namespace aos

} // namespace px

// ===== 원본: physx/include/foundation/PxBasicTemplates.h
namespace px
{
	template <typename A>
	struct PxEqual
	{
		EHDI bool operator()(const A& a, const A& b) const
		{
			return a == b;
		}
	};

	template <typename A>
	struct PxLess
	{
		EHDI bool operator()(const A& a, const A& b) const
		{
			return a < b;
		}
	};

	template <typename A>
	struct PxGreater
	{
		EHDI bool operator()(const A& a, const A& b) const
		{
			return a > b;
		}
	};

	template <class F, class S>
	class PxPair
	{
	public:
		F first;
		S second;
		EHDI PxPair() : first(F()), second(S())
		{
		}
		EHDI PxPair(const F& f, const S& s) : first(f), second(s)
		{
		}
		EHDI PxPair(const PxPair& p) : first(p.first), second(p.second)
		{
		}
		EHDI PxPair& operator=(const PxPair& p)
		{
			first = p.first;
			second = p.second;
			return *this;
		}
		EHDI bool operator==(const PxPair& p) const
		{
			return first == p.first && second == p.second;
		}
		EHDI bool operator<(const PxPair& p) const
		{
			if (first < p.first)
				return true;
			else
				return !(p.first < first) && (second < p.second);
		}
	};

	template <unsigned int A>
	struct PxLogTwo
	{
		static const unsigned int value = PxLogTwo<(A >> 1)>::value + 1;
	};
	template <>
	struct PxLogTwo<1>
	{
		static const unsigned int value = 0;
	};

	template <typename T>
	struct PxUnConst
	{
		typedef T Type;
	};
	template <typename T>
	struct PxUnConst<const T>
	{
		typedef T Type;
	};

	template <typename T>
	EHDI T PxPointerOffset(void* p, ptrdiff_t offset)
	{
		return reinterpret_cast<T>(reinterpret_cast<char*>(p) + offset);
	}
	template <typename T>
	EHDI T PxPointerOffset(const void* p, ptrdiff_t offset)
	{
		return reinterpret_cast<T>(reinterpret_cast<const char*>(p) + offset);
	}

	template <class T>
	EHDI void PxSwap(T& x, T& y)
	{
		const T tmp = x;
		x = y;
		y = tmp;
	}

} // namespace px

// ===== 원본: physx/include/foundation/PxUtilities.h
namespace px
{
EHDI char PxLittleEndian()
{
	int i = 1;
	return *(reinterpret_cast<char*>(&i));
}

// PT: checked casts
EHD PxU32 PxTo32(PxU64 value)
{
	E_ASSERT(value <= 0xffffffff);
	return PxU32(value);
}
EHD PxU32 PxToU32(PxI32 value)
{
	E_ASSERT(value >= 0);
	return PxU32(value);
}
EHD PxU16 PxTo16(PxU32 value)
{
	E_ASSERT(value <= 0xffff);
	return PxU16(value);
}
EHD PxU8 PxTo8(PxU16 value)
{
	E_ASSERT(value <= 0xff);
	return PxU8(value);
}
EHD PxU8 PxTo8(PxU32 value)
{
	E_ASSERT(value <= 0xff);
	return PxU8(value);
}
EHD PxU8 PxTo8(PxI32 value)
{
	E_ASSERT(value <= 0xff);
	E_ASSERT(value >= 0);
	return PxU8(value);
}
EHD PxI8 PxToI8(PxU32 value)
{
	E_ASSERT(value <= 0x7f);
	return PxI8(value);
}

//! \cond
/*!
Get number of elements in array
*/
template <typename T, size_t N>
EHDV char (&PxArraySizeHelper(T (&array)[N]))[N];
#define EPX_ARRAY_SIZE(_array) (sizeof(px::PxArraySizeHelper(_array)))
//! \endcond

/*!
Sort two elements using operator<

On return x will be the smaller of the two
*/
template <class T>
EHD void PxOrder(T& x, T& y)
{
	if(y < x)
		PxSwap(x, y);
}

// most architectures can do predication on real comparisons, and on VMX, it matters

EHD void PxOrder(PxReal& x, PxReal& y)
{
	PxReal newX = PxMin(x, y);
	PxReal newY = PxMax(x, y);
	x = newX;
	y = newY;
}

/*!
Sort two elements using operator< and also keep order
of any extra data
*/
template <class T, class E1>
EHD void PxOrder(T& x, T& y, E1& xe1, E1& ye1)
{
	if(y < x)
	{
		swap(x, y);
		swap(xe1, ye1);
	}
}

__attribute__((noreturn))
    EHDI void PxDebugBreak()
{
	__builtin_trap();
}

} // namespace px

// ===== 원본: physx/include/foundation/PxBitAndData.h
namespace px
{

template <typename storageType, storageType bitMask>
class PxBitAndDataT
{
  public:
	EHD PxBitAndDataT(const PxEMPTY)
	{
	}
	EHD PxBitAndDataT() : mData(0)
	{
	}
	EHD PxBitAndDataT(storageType data, bool bit = false)
	{
		mData = bit ? storageType(data | bitMask) : data;
	}

	EHD operator storageType() const
	{
		return storageType(mData & ~bitMask);
	}
	EHD void setBit()
	{
		mData |= bitMask;
	}
	EHD void clearBit()
	{
		mData &= ~bitMask;
	}
	EHD storageType isBitSet() const
	{
		return storageType(mData & bitMask);
	}

  protected:
	storageType mData;
};
typedef PxBitAndDataT<PxU8, 0x80> PxBitAndByte;
typedef PxBitAndDataT<PxU16, 0x8000> PxBitAndWord;
typedef PxBitAndDataT<PxU32, 0x80000000> PxBitAndDword;

} // namespace px

// ===== 원본: physx/include/geometry/PxMeshScale.h
/** \brief Minimum allowed absolute magnitude for each of mesh scale's components (x,y,z).
	\note Only positive scale values are allowed for convex meshes. */
#define EPX_MESH_SCALE_MIN 1e-6f

/** \brief Maximum allowed absolute magnitude for each of mesh scale's components (x,y,z).
	\note Only positive scale values are allowed for convex meshes. */
#define EPX_MESH_SCALE_MAX 1e6f

namespace px
{

/**
\brief A class expressing a nonuniform scaling transformation.

The scaling is along arbitrary axes that are specified by PxMeshScale::rotation. Specifically, PxMeshScale::rotation
describes the rotation from the scaling-axes frame to the mesh-local frame, i.e. PxMeshScale::rotation.rotate(v) transforms
the coordinates of vertex v from the mesh-local frame to the scaling-axes frame.

\note	Negative scale values are supported for PxTriangleMeshGeometry
		with absolute values for each component within [EPX_MIN_ABS_MESH_SCALE, EPX_MAX_ABS_MESH_SCALE] range.
		Negative scale causes a reflection around the specified axis, in addition PhysX will flip the normals
		for mesh triangles when scale.x*scale.y*scale.z < 0.
\note	Only positive scale values are supported for PxConvexMeshGeometry
		with values for each component within [EPX_MIN_ABS_MESH_SCALE, EPX_MAX_ABS_MESH_SCALE] range).

\see PxConvexMeshGeometry PxTriangleMeshGeometry
*/
class PxMeshScale
{
public:
	/**
	\brief Constructor initializes to identity scale.
	*/
	EHD PxMeshScale(): scale(1.0f), rotation(PxIdentity) 
	{
	}

	/**
	\brief Constructor from scalar.
	*/
	explicit EHD PxMeshScale(PxReal r): scale(r), rotation(PxIdentity) 
	{
	}

	/**
	\brief Constructor to initialize to arbitrary scale and identity scale rotation.
	*/
	EHD PxMeshScale(const PxVec3& s)
	{
		scale = s;
		rotation = PxQuat(PxIdentity);
	}

	/**
	\brief Constructor to initialize to arbitrary scaling.
	*/
	EHD PxMeshScale(const PxVec3& s, const PxQuat& r)
	{
		E_ASSERT(r.isUnit());
		scale = s;
		rotation = r;
	}

	/**
	\brief Returns true if the scaling is an identity transformation.
	*/
	EHD bool isIdentity()	const
	{
		return (scale.x == 1.0f && scale.y == 1.0f && scale.z == 1.0f);
	}

	/**
	\brief Returns the inverse of this scaling transformation.
	*/
	EHD  PxMeshScale getInverse() const 
	{
		return PxMeshScale(PxVec3(1.0f/scale.x, 1.0f/scale.y, 1.0f/scale.z), rotation);
	}

	/**
	\brief Converts this transformation to a 3x3 matrix representation.
	*/
	EHD PxMat33 toMat33() const 
	{
		PxMat33 rot(rotation);
		PxMat33 trans = rot.getTranspose();
		trans.column0 *= scale[0];
		trans.column1 *= scale[1];
		trans.column2 *= scale[2];
		return trans * rot;
	}

	/**
	\brief Returns true if combination of negative scale components will cause the triangle normal to flip. The SDK will flip the normals internally.
	*/
	EHD bool hasNegativeDeterminant() const
	{
		return (scale.x * scale.y * scale.z < 0.0f);
	}

	EHDI PxVec3		transform(const PxVec3& v) const
	{
		return rotation.rotateInv(scale.multiply(rotation.rotate(v)));
	}

	EHDI bool		isValidForTriangleMesh() const
	{
		PxVec3 absXYZ = scale.abs();
		return (absXYZ.maxElement() <= EPX_MESH_SCALE_MAX) && (absXYZ.minElement() >= EPX_MESH_SCALE_MIN);
	}

	EHDI bool		isValidForConvexMesh() const
	{
		return (scale.maxElement() <= EPX_MESH_SCALE_MAX) && (scale.minElement() >= EPX_MESH_SCALE_MIN);
	}

	PxVec3	scale;		//!< A nonuniform scaling
	PxQuat	rotation;	//!< The orientation of the scaling axes
};

} // namespace px

#include "core/contact/px/gu_geometry.inc"

// ===== 원본: physx/include/geomutils/PxContactPoint.h
namespace px
{

	struct PxContactPoint
	{
		/**
		\brief The normal of the contacting surfaces at the contact point.

		For two shapes s0 and s1, the normal points in the direction that s0 needs to move in to resolve the contact with s1.
		*/
		alignas(16) PxVec3	normal;

		/**
		\brief The separation of the shapes at the contact point. A negative separation denotes a penetration.
		*/
		PxReal	separation;

		/**
		\brief The point of contact between the shapes, in world space. 
		*/
		alignas(16) PxVec3	point;	

		/**
		\brief The max impulse permitted at this point
		*/
		PxReal maxImpulse;

		alignas(16) PxVec3 targetVel;

		/**
		\brief The static friction coefficient
		*/
		PxReal staticFriction;

		/**
		\brief Material flags for this contact (eDISABLE_FRICTION, eDISABLE_STRONG_FRICTION). \see PxMaterialFlag
		*/
		PxU8 materialFlags;

		/**
		\brief The surface index of shape 1 at the contact point. This is used to identify the surface material.

		\note This field is only supported by triangle meshes and heightfields, else it will be set to PXC_CONTACT_NO_FACE_INDEX.
		*/
		PxU32   internalFaceIndex1;

		/**
		\brief The dynamic friction coefficient
		*/
		PxReal dynamicFriction;

		/**
		\brief The restitution coefficient
		*/
		PxReal restitution;

		/**
		\brief Damping coefficient (for compliant contacts)
		*/
		PxReal damping;
	};

} // namespace px

// ===== 원본: physx/include/geomutils/PxContactBuffer.h
namespace px
{

	class PxContactBuffer
	{
	public:

		static const PxU32 MAX_CONTACTS = 256;

		PxContactPoint	contacts[MAX_CONTACTS];
		PxU32			count;
		PxU32			pad;

		EHD void reset()
		{
			count = 0;
		}

		EHD bool contact(const PxVec3& worldPoint, const PxVec3& worldNormalIn, PxReal separation, PxU32 faceIndex1 = PXC_CONTACT_NO_FACE_INDEX)
		{
			E_ASSERT(PxAbs(worldNormalIn.magnitude()-1)<1e-3f);

			if(count>=MAX_CONTACTS)
				return false;

			PxContactPoint& p	= contacts[count++];
			p.normal			= worldNormalIn;
			p.point				= worldPoint;
			p.separation		= separation;
			p.internalFaceIndex1= faceIndex1;
			return true;
		}

		EHD bool contact(const PxContactPoint& pt)
		{
			if(count>=MAX_CONTACTS)
				return false;
			contacts[count++] = pt;
			return true;
		}

		EHD PxContactPoint* contact()
		{
			if(count>=MAX_CONTACTS)
				return NULL;
			return &contacts[count++];
		}
	};

} // namespace px

// ===== 원본: physx/source/geomutils/include/GuCenterExtents.h
namespace px
{
namespace Gu
{
	class CenterExtents : public px::PxUserAllocated
	{
		public:
		EHD				CenterExtents()						{}
		EHD				CenterExtents(const PxBounds3& b)	{ mCenter = b.getCenter();	mExtents = b.getExtents();	}
		EHD				~CenterExtents()					{}

		EHD	void		getMin(PxVec3& min)		const		{ min = mCenter - mExtents;					}
		EHD	void		getMax(PxVec3& max)		const		{ max = mCenter + mExtents;					}

		EHD	float		getMin(PxU32 axis)		const		{ return mCenter[axis] - mExtents[axis];	}
		EHD	float		getMax(PxU32 axis)		const		{ return mCenter[axis] + mExtents[axis];	}

		EHD	PxVec3		getMin()				const		{ return mCenter - mExtents;				}
		EHD	PxVec3		getMax()				const		{ return mCenter + mExtents;				}

		EHD	void		setMinMax(const PxVec3& min, const PxVec3& max)
									{
										mCenter = (max + min)*0.5f;
										mExtents = (max - min)*0.5f;
									}

		EHD	PxU32		isInside(const CenterExtents& box)	const
									{
										if(box.getMin(0)>getMin(0))	return 0;
										if(box.getMin(1)>getMin(1))	return 0;
										if(box.getMin(2)>getMin(2))	return 0;
										if(box.getMax(0)<getMax(0))	return 0;
										if(box.getMax(1)<getMax(1))	return 0;
										if(box.getMax(2)<getMax(2))	return 0;
										return 1;
									}

		EHD	void		setEmpty()
									{
										mExtents = PxVec3(-EPX_MAX_BOUNDS_EXTENTS);
									}

		EHD	bool		isEmpty()	const
									{
										E_ASSERT(isValid());
										return mExtents.x<0.0f;
									}

		EHD	bool		isFinite()	const
									{
										return mCenter.isFinite() && mExtents.isFinite();
									}

		EHD	bool		isValid()	const
									{
										const PxVec3& c = mCenter;
										const PxVec3& e = mExtents;
										return (c.isFinite() && e.isFinite() && (((e.x >= 0.0f) && (e.y >= 0.0f) && (e.z >= 0.0f)) ||
																				((e.x == -EPX_MAX_BOUNDS_EXTENTS) &&
																				(e.y == -EPX_MAX_BOUNDS_EXTENTS) &&
																				(e.z == -EPX_MAX_BOUNDS_EXTENTS))));
									}

		EHD	PxBounds3	transformFast(const PxMat33& matrix)	const
									{
										E_ASSERT(isValid());
										return PxBounds3::basisExtent(matrix * mCenter, matrix, mExtents);
									}

						PxVec3		mCenter;
						PxVec3		mExtents;
	};

	//! A padded version of CenterExtents, to safely load its data using SIMD
	class CenterExtentsPadded : public CenterExtents
	{
	public:
		EHD CenterExtentsPadded()	{}
		EHD ~CenterExtentsPadded()	{}
		PxU32	padding;
	};
	E_STATIC_ASSERT(sizeof(CenterExtentsPadded) == 7*4);

}

}

// ===== 원본: physx/source/common/src/CmScaling.h
namespace px
{
namespace Cm
{
	// PT: same as PxMeshScale::toMat33() but faster
	EHD PxMat33 toMat33(const PxMeshScale& meshScale)
	{
		const PxMat33Padded rot(meshScale.rotation);

		PxMat33 trans = rot.getTranspose();
		trans.column0 *= meshScale.scale[0];
		trans.column1 *= meshScale.scale[1];
		trans.column2 *= meshScale.scale[2];
		return trans * rot;
	}

	// class that can perform scaling fast.  Relatively large size, generated from PxMeshScale on demand.
	// CS: I've removed most usages of this class, because most of the time only one-way transform is needed.
	// If you only need a temporary FastVertex2ShapeScaling, setup your transform as PxMat34Legacy and use
	// normal matrix multiplication or a transform() overload to convert points and bounds between spaces.
	class FastVertex2ShapeScaling
	{
	public:
		EHDI FastVertex2ShapeScaling()
		{
			//no scaling by default:
			vertex2ShapeSkew = PxMat33(PxIdentity);
			shape2VertexSkew = PxMat33(PxIdentity);
			mFlipNormal = false;
		}

		EHDI explicit FastVertex2ShapeScaling(const PxMeshScale& scale)
		{
			init(scale);
		}

		EHDI FastVertex2ShapeScaling(const PxVec3& scale, const PxQuat& rotation)
		{
			init(scale, rotation);
		}

		EHDI void init(const PxMeshScale& scale)
		{
			init(scale.scale, scale.rotation);
		}

		EHDI void setIdentity()
		{
			vertex2ShapeSkew = PxMat33(PxIdentity);
			shape2VertexSkew = PxMat33(PxIdentity);
			mFlipNormal = false;
		}

		EHDI void init(const PxVec3& scale, const PxQuat& rotation)
		{
			// TODO: may want to optimize this for cases where we have uniform or axis aligned scaling!
			// That would introduce branches and it's unclear to me whether that's faster than just doing the math.
			// Lazy computation would be another option, at the cost of introducing even more branches.

			const PxMat33Padded R(rotation);
			vertex2ShapeSkew = R.getTranspose();
			const PxMat33 diagonal = PxMat33::createDiagonal(scale);
			vertex2ShapeSkew = vertex2ShapeSkew * diagonal;
			vertex2ShapeSkew = vertex2ShapeSkew * R;

			/*
			The inverse, is, explicitly:
			shape2VertexSkew.setTransposed(R);
			shape2VertexSkew.multiplyDiagonal(PxVec3(1.0f/scale.x, 1.0f/scale.y, 1.0f/scale.z));
			shape2VertexSkew *= R;

			It may be competitive to compute the inverse -- though this has a branch in it:
			*/
			
			shape2VertexSkew = vertex2ShapeSkew.getInverse();			

			mFlipNormal = ((scale.x * scale.y * scale.z) < 0.0f);
		}

		EHD void flipNormal(PxVec3& v1, PxVec3& v2) const
		{
			if (mFlipNormal)
			{
				PxVec3 tmp = v1; v1 = v2; v2 = tmp;
			}
		}

		EHD PxVec3 operator*  (const PxVec3& src) const
		{
			return vertex2ShapeSkew * src;
		}

		EHD PxVec3 operator%  (const PxVec3& src) const
		{
			return shape2VertexSkew * src;
		}

		EHD const PxMat33& getVertex2ShapeSkew() const
		{
			return vertex2ShapeSkew;
		}

		EHD const PxMat33& getShape2VertexSkew() const
		{
			return shape2VertexSkew;
		}

		EHDI PxMat34 getVertex2WorldSkew(const PxMat34& shape2world) const
		{
			const PxMat34 vertex2worldSkew = shape2world * getVertex2ShapeSkew();
			//vertex2worldSkew = shape2world * [vertex2shapeSkew, 0]
			//[aR at] * [bR bt] = [aR * bR		aR * bt + at]  NOTE: order of operations important so it works when this ?= left ?= right.
			return vertex2worldSkew;
		}

		EHDI PxMat34 getWorld2VertexSkew(const PxMat34& shape2world) const
		{
			//world2vertexSkew = shape2vertex * invPQ(shape2world)
			//[aR 0] * [bR' -bR'*bt] = [aR * bR'		-aR * bR' * bt + 0]

			const PxMat33 rotate( shape2world[0], shape2world[1], shape2world[2] );
			const PxMat33 M = getShape2VertexSkew() * rotate.getTranspose();
			return PxMat34(M[0], M[1], M[2], -M * shape2world[3]);			
		}

		//! Transforms a shape space OBB to a vertex space OBB.  All 3 params are in and out.
		EHDI void transformQueryBounds(PxVec3& center, PxVec3& extents, PxMat33& basis)	const
		{
			basis.column0 = shape2VertexSkew * (basis.column0 * extents.x);
			basis.column1 = shape2VertexSkew * (basis.column1 * extents.y);
			basis.column2 = shape2VertexSkew * (basis.column2 * extents.z);

			center = shape2VertexSkew * center;
			extents = PxOptimizeBoundingBox(basis);
		}

		EHDI void transformPlaneToShapeSpace(const PxVec3& nIn, const PxReal dIn, PxVec3& nOut, PxReal& dOut)	const
		{
			const PxVec3 tmp = shape2VertexSkew.transformTranspose(nIn);
			const PxReal denom = 1.0f / tmp.magnitude();
			nOut = tmp * denom;
			dOut = dIn * denom;
		}

		EHD bool flipsNormal() const { return mFlipNormal; }

	private:
		PxMat33 vertex2ShapeSkew;
		PxMat33 shape2VertexSkew;
		bool mFlipNormal;
	};

	EHD void getScaledVertices(PxVec3* v, const PxVec3& v0, const PxVec3& v1, const PxVec3& v2, bool idtMeshScale, const Cm::FastVertex2ShapeScaling& scaling)
	{
		if(idtMeshScale)
		{
			v[0] = v0;
			v[1] = v1;
			v[2] = v2;
		}
		else
		{
			const PxI32 winding = scaling.flipsNormal() ? 1 : 0;
			v[0] = scaling * v0;
			v[1+winding] = scaling * v1;
			v[2-winding] = scaling * v2;
		}
	}

} // namespace Cm


EHDI PxMat34 operator*(const PxTransform& transform, const PxMeshScale& scale) 
{
	const PxMat33Padded tmp(transform.q);

	return PxMat34(tmp * Cm::toMat33(scale), transform.p);
}

EHDI PxMat34 operator*(const PxMeshScale& scale, const PxTransform& transform) 
{
	const PxMat33 scaleMat = Cm::toMat33(scale);
	const PxMat33Padded t(transform.q);
	const PxMat33 r = scaleMat * t;
	const PxVec3 p = scaleMat * transform.p;
	return PxMat34(r, p);
}

EHDI PxMat34 operator*(const PxMat34& transform, const PxMeshScale& scale) 
{
	return PxMat34(transform.m * Cm::toMat33(scale), transform.p);
}

EHDI PxMat34 operator*(const PxMeshScale& scale, const PxMat34& transform) 
{
	const PxMat33 scaleMat = Cm::toMat33(scale);
	return PxMat34(scaleMat * transform.m, scaleMat * transform.p);
}

}

// ===== 원본: physx/source/common/src/CmTransformUtils.h
namespace
{

using namespace px::aos;

// V3PrepareCross would help here, but it's not on all platforms yet...

EHD void transformFast(const FloatVArg wa, const Vec3VArg va, const Vec3VArg pa, 
							   const FloatVArg wb, const Vec3VArg vb, const Vec3VArg pb, 
							   FloatV& wo, Vec3V& vo, Vec3V& po)
{
	wo = FSub(FMul(wa, wb), V3Dot(va, vb));
	vo = V3ScaleAdd(va, wb, V3ScaleAdd(vb, wa, V3Cross(va, vb)));

	const Vec3V t1 = V3Scale(pb, FScaleAdd(wa, wa, FLoad(-0.5f)));
	const Vec3V t2 = V3ScaleAdd(V3Cross(va, pb), wa, t1);
	const Vec3V t3 = V3ScaleAdd(va, V3Dot(va, pb), t2);

	po = V3ScaleAdd(t3, FLoad(2.f), pa);
}

EHD void transformInvFast(const FloatVArg wa, const Vec3VArg va, const Vec3VArg pa, 
								      const FloatVArg wb, const Vec3VArg vb, const Vec3VArg pb, 
								      FloatV& wo, Vec3V& vo, Vec3V& po)
{
	wo = FScaleAdd(wa, wb, V3Dot(va, vb));
	vo = V3NegScaleSub(va, wb, V3ScaleAdd(vb, wa, V3Cross(vb, va)));

	const Vec3V pt = V3Sub(pb, pa);
	const Vec3V t1 = V3Scale(pt, FScaleAdd(wa, wa, FLoad(-0.5f)));
	const Vec3V t2 = V3ScaleAdd(V3Cross(pt, va), wa, t1);
	const Vec3V t3 = V3ScaleAdd(va, V3Dot(va, pt), t2);
	po = V3Add(t3,t3);
}

}

namespace px
{
namespace Cm
{

// PT: actor2World * shape2Actor
EHD void getStaticGlobalPoseAligned(const PxTransform& actor2World, const PxTransform& shape2Actor, PxTransform& outTransform)
{
	using namespace aos;

	E_ASSERT((size_t(&actor2World)&15) == 0);
	E_ASSERT((size_t(&shape2Actor)&15) == 0);
	E_ASSERT((size_t(&outTransform)&15) == 0);

	const Vec3V actor2WorldPos = V3LoadA(actor2World.p);
	const QuatV actor2WorldRot = QuatVLoadA(&actor2World.q.x);

	const Vec3V shape2ActorPos = V3LoadA(shape2Actor.p);
	const QuatV shape2ActorRot = QuatVLoadA(&shape2Actor.q.x);
	
	Vec3V v,p;
	FloatV w; 

	transformFast(V4GetW(actor2WorldRot), Vec3V_From_Vec4V(actor2WorldRot), actor2WorldPos,
				  V4GetW(shape2ActorRot), Vec3V_From_Vec4V(shape2ActorRot), shape2ActorPos,
				  w, v, p);

	V3StoreA(p, outTransform.p);
	V4StoreA(V4SetW(v,w), &outTransform.q.x);
}

// PT: body2World * body2Actor.getInverse() * shape2Actor
EHD void getDynamicGlobalPoseAligned(const PxTransform& body2World, const PxTransform& shape2Actor, const PxTransform& body2Actor, PxTransform& outTransform)
{
	E_ASSERT((size_t(&body2World)&15) == 0);
	E_ASSERT((size_t(&shape2Actor)&15) == 0);
	E_ASSERT((size_t(&body2Actor)&15) == 0);
	E_ASSERT((size_t(&outTransform)&15) == 0);

	using namespace aos;

	const Vec3V shape2ActorPos = V3LoadA(shape2Actor.p);
	const QuatV shape2ActorRot = QuatVLoadA(&shape2Actor.q.x);

	const Vec3V body2ActorPos = V3LoadA(body2Actor.p);
	const QuatV body2ActorRot = QuatVLoadA(&body2Actor.q.x);

	const Vec3V body2WorldPos = V3LoadA(body2World.p);
	const QuatV body2WorldRot = QuatVLoadA(&body2World.q.x);

	Vec3V v1, p1, v2, p2;
	FloatV w1, w2; 

	transformInvFast(V4GetW(body2ActorRot), Vec3V_From_Vec4V(body2ActorRot), body2ActorPos,
					 V4GetW(shape2ActorRot), Vec3V_From_Vec4V(shape2ActorRot), shape2ActorPos,
					 w1, v1, p1);

	transformFast(V4GetW(body2WorldRot), Vec3V_From_Vec4V(body2WorldRot), body2WorldPos,
			      w1, v1, p1,
			      w2, v2, p2);

	V3StoreA(p2, outTransform.p);
	V4StoreA(V4SetW(v2, w2), &outTransform.q.x);
}

}
}

// ===== 원본: physx/source/geomutils/src/convex/GuConvexMeshData.h
// Data definition

namespace px
{
namespace Gu
{
	struct BigConvexRawData;

	struct HullPolygonData
	{
		// PT: this structure won't be allocated with EPX_NEW because polygons aren't allocated alone (with a dedicated alloc).
		// Instead they are part of a unique allocation/buffer containing all data for the ConvexHullData class (polygons, followed by
		// hull vertices, edge data, etc). As a result, ctors for embedded classes like PxPlane won't be called.

		PxPlane	mPlane;			//!< Plane equation for this polygon	//Could drop 4th elem as it can be computed from any vertex as: d = - p.dot(n);
		PxU16	mVRef8;			//!< Offset of vertex references in hull vertex data (CS: can we assume indices are tightly packed and offsets are ascending?? DrawObjects makes and uses this assumption)
		PxU8	mNbVerts;		//!< Number of vertices/edges in the polygon
		PxU8	mMinIndex;		//!< Index of the polygon vertex that has minimal projection along this plane's normal.

		EHD	PxReal getMin(const PxVec3*  hullVertices) const	//minimum of projection of the hull along this plane normal
		{ 
			return mPlane.n.dot(hullVertices[mMinIndex]);
		}

		EHD	PxReal getMax() const		{ return -mPlane.d; }	//maximum of projection of the hull along this plane normal
	};

	EHD void flipData(Gu::HullPolygonData& data)
	{
		flip(data.mPlane.n.x);
		flip(data.mPlane.n.y);
		flip(data.mPlane.n.z);
		flip(data.mPlane.d);
		flip(data.mVRef8);
	}
	// PT: if this one breaks, please make sure the 'flipData' function is properly updated.
	E_STATIC_ASSERT(sizeof(Gu::HullPolygonData) == 20);

// TEST_INTERNAL_OBJECTS
	struct InternalObjectsData
	{
		PxVec3	mInternalExtents;
		PxReal	mInternalRadius;

		EHD	void reset()
		{
			mInternalExtents = PxVec3(0.0f);
			mInternalRadius = 0.0f;
		}
	};
	E_STATIC_ASSERT(sizeof(Gu::InternalObjectsData) == 16);
	// PT: ensure that mInternalExtents is not the last member of InternalObjectsData, i.e. it is safe to load 4 bytes after mInternalExtents
	E_STATIC_ASSERT(E_OFFSET_OF(InternalObjectsData, mInternalExtents)+sizeof(InternalObjectsData::mInternalExtents) + 4 <= sizeof(InternalObjectsData));
//~TEST_INTERNAL_OBJECTS

	struct ConvexHullData
	{
		// PT: WARNING: bounds must be followed by at least 32bits of data for safe SIMD loading
		CenterExtents		mAABB;				//!< bounds TODO: compute this on the fly from first 6 vertices in the vertex array.  We'll of course need to sort the most extreme ones to the front.
		PxVec3				mCenterOfMass;		//in local space of mesh!

		// PT: WARNING: mNbHullVertices *must* appear before mBigConvexRawData for ConvX to be able to do "big raw data" surgery
	
		PxBitAndWord		mNbEdges;			//!<the highest bit indicate whether we have grb data, the other 15 bits indicate the number of edges in this convex hull

		PxU8				mNbHullVertices;	//!< Number of vertices in the convex hull
		PxU8				mNbPolygons;		//!< Number of planar polygons composing the hull

		HullPolygonData*	mPolygons;			//!< Array of mNbPolygons structures
		BigConvexRawData*	mBigConvexRawData;	//!< Hill climbing data, only for large convexes! else NULL.

		// SDF data
		SDF*				mSdfData;

// TEST_INTERNAL_OBJECTS
		InternalObjectsData	mInternal;
//~TEST_INTERNAL_OBJECTS

		EHD ConvexHullData(const PxEMPTY) : mNbEdges(PxEmpty)
		{
		}

		EHD ConvexHullData()
		{
		}

		EHD	const CenterExtentsPadded& getPaddedBounds() const
		{
			// PT: see compile-time assert at the end of file
			return static_cast<const CenterExtentsPadded&>(mAABB);
		}

		EHD	const PxVec3* getHullVertices()	const	//!< Convex hull vertices
		{
			const char* tmp = reinterpret_cast<const char*>(mPolygons);
			tmp += sizeof(Gu::HullPolygonData) * mNbPolygons;
			return reinterpret_cast<const PxVec3*>(tmp);
		}

		EHD	const PxU8* getFacesByEdges8()	const	//!< for each edge, gives 2 adjacent polygons; used by convex-convex code to come up with all the convex' edge normals.  
		{
			const char* tmp = reinterpret_cast<const char*>(mPolygons);
			tmp += sizeof(Gu::HullPolygonData) * mNbPolygons;
			tmp += sizeof(PxVec3) * mNbHullVertices; 
			return reinterpret_cast<const PxU8*>(tmp);
		}

		EHD	const PxU8* getFacesByVertices8()	const	//!< for each edge, gives 2 adjacent polygons; used by convex-convex code to come up with all the convex' edge normals.  
		{
			const char* tmp = reinterpret_cast<const char*>(mPolygons);
			tmp += sizeof(Gu::HullPolygonData) * mNbPolygons;
			tmp += sizeof(PxVec3) * mNbHullVertices;
			tmp += sizeof(PxU8) * mNbEdges * 2;
			return reinterpret_cast<const PxU8*>(tmp);
		}

		//If we don't build the convex hull with grb data, we will return NULL pointer
		EHD	const PxU16*	getVerticesByEdges16()	const	//!< Vertex indices indexed by unique edges
		{
			if (mNbEdges.isBitSet())
			{
				const char* tmp = reinterpret_cast<const char*>(mPolygons);
				tmp += sizeof(Gu::HullPolygonData) * mNbPolygons;
				tmp += sizeof(PxVec3) * mNbHullVertices;
				tmp += sizeof(PxU8) * mNbEdges * 2;
				tmp += sizeof(PxU8) * mNbHullVertices * 3;
				return reinterpret_cast<const PxU16*>(tmp);
			}
			return NULL;
		}

		EHD	const PxU8*	getVertexData8()	const	//!< Vertex indices indexed by hull polygons
		{
			const char* tmp = reinterpret_cast<const char*>(mPolygons);
			tmp += sizeof(Gu::HullPolygonData) * mNbPolygons;
			tmp += sizeof(PxVec3) * mNbHullVertices;
			tmp += sizeof(PxU8) * mNbEdges * 2;
			tmp += sizeof(PxU8) * mNbHullVertices * 3;
			if (mNbEdges.isBitSet())
				tmp += sizeof(PxU16) * mNbEdges * 2;
			return reinterpret_cast<const PxU8*>(tmp);
		}

		EHD bool checkExtentRadiusRatio()	const
		{
			const PxReal maxR = mInternal.mInternalExtents.maxElement();
			const PxReal minR = mInternal.mInternalRadius;
			const PxReal ratio = maxR/minR;

			return ratio < 100.f;
		}

	};
	E_STATIC_ASSERT(sizeof(Gu::ConvexHullData) == 80);

	// PT: 'getPaddedBounds()' is only safe if we make sure the bounds member is followed by at least 32bits of data
	E_STATIC_ASSERT(E_OFFSET_OF(Gu::ConvexHullData, mCenterOfMass)>=E_OFFSET_OF(Gu::ConvexHullData, mAABB)+4);

} // namespace Gu

}

//#pragma EPX_POP_PACK

// ===== 원본: physx/source/geomutils/src/convex/GuBigConvexData.h
namespace px
{

class BigConvexDataBuilder;
class PxcHillClimb;
class BigConvexData;

// Data

namespace Gu
{

struct Valency
{
	PxU16		mCount;
	PxU16		mOffset;
};
E_STATIC_ASSERT(sizeof(Gu::Valency) == 4);

struct BigConvexRawData
{
	// Support vertex map
	PxU16		mSubdiv;		// "Gaussmap" subdivision
	PxU16		mNbSamples;		// Total #samples in gaussmap PT: this is not even needed at runtime!

	PxU8*		mSamples;
	EHD const PxU8*	getSamples2()	const
	{
		return mSamples + mNbSamples;
	}
	//~Support vertex map

	// Valencies data
	PxU32			mNbVerts;		//!< Number of vertices
	PxU32			mNbAdjVerts;	//!< Total number of adjacent vertices  ### PT: this is useless at runtime and should not be stored here
	Gu::Valency*	mValencies;		//!< A list of mNbVerts valencies (= number of neighbors)
	PxU8*			mAdjacentVerts;	//!< List of adjacent vertices
	//~Valencies data
};
E_STATIC_ASSERT(sizeof(Gu::BigConvexRawData) == 40);

} // namespace Gu

}

// ===== 원본: physx/source/geomutils/src/convex/GuCubeIndex.h
namespace px
{

	enum CubeIndex
	{
		CUBE_RIGHT,
		CUBE_LEFT,
		CUBE_TOP,
		CUBE_BOTTOM,
		CUBE_FRONT,
		CUBE_BACK,

		CUBE_FORCE_DWORD	= 0x7fffffff
	};

	/*
		It's pretty straightforwards in concept (though the execution in hardware is
		a bit crufty and complex). You use a 3D texture coord to look up a texel in
		a cube map. First you find which of the axis has the largest value (i.e.
		X,Y,Z), and then the sign of that axis decides which face you are going to
		use. Which is why the faces are called +X, -X, +Y, -Y, +Z, -Z - after their
		principle axis. Then you scale the vector so that the largest value is +/-1.
		Then use the other two as 2D coords to look up your texel (with a 0.5 scale
		& offset).

		For example, vector (0.4, -0.2, -0.5). Largest value is the Z axis, and it's
		-ve, so we're reading from the -Z map. Scale so that this Z axis is +/-1,
		and you get the vector (0.8, -0.4, -1.0). So now use the other two values to
		look up your texel. So we look up texel (0.8, -0.4). The scale & offset move
		the -1->+1 range into the usual 0->1 UV range, so we actually look up texel
		(0.9, 0.3). The filtering is extremely complex, especially where three maps
		meet, but that's a hardware problem :-)
	*/

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 *	Cubemap lookup function.
	 *
	 *	To transform returned uvs into mapping coordinates :
	 *	u += 1.0f;	u *= 0.5f;
	 *	v += 1.0f;	v *= 0.5f;
	 *
	 *	\fn			CubemapLookup(const PxVec3& direction, float& u, float& v)
	 *	\param		direction	[in] a direction vector
	 *	\param		u			[out] impact coordinate on the unit cube, in [-1,1]
	 *	\param		v			[out] impact coordinate on the unit cube, in [-1,1]
	 *	\return		cubemap texture index
	 */
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	EHDI CubeIndex		CubemapLookup(const PxVec3& direction, float& u, float& v);

	EHDI PxU32 ComputeCubemapOffset(const PxVec3& dir, PxU32 subdiv)
	{
		float u,v;
		const CubeIndex CI = CubemapLookup(dir, u, v);

		// Remap to [0, subdiv[
		const float Coeff = 0.5f * float(subdiv-1);
		u += 1.0f;	u *= Coeff;
		v += 1.0f;	v *= Coeff;

		// Compute offset
		return PxU32(CI)*(subdiv*subdiv) + PxU32(u)*subdiv + PxU32(v);
	}


	EHDI PxU32 ComputeCubemapNearestOffset(const PxVec3& dir, PxU32 subdiv)
	{
		float u,v;
		const CubeIndex CI = CubemapLookup(dir, u, v);

		// Remap to [0, subdiv]
		const float Coeff = 0.5f * float(subdiv-1);
		u += 1.0f;	u *= Coeff;
		v += 1.0f;	v *= Coeff;

		// Compute offset
		return PxU32(CI)*(subdiv*subdiv) + PxU32(u + 0.5f)*subdiv + PxU32(v + 0.5f);
	}


	EHDI CubeIndex CubemapLookup(const PxVec3& direction, float& u, float& v)
	{
		const PxU32* binary = reinterpret_cast<const PxU32*>(&direction.x);

		const PxU32 absPx = binary[0] & ~EPX_SIGN_BITMASK;
		const PxU32 absNy = binary[1] & ~EPX_SIGN_BITMASK;
		const PxU32 absNz = binary[2] & ~EPX_SIGN_BITMASK;

		PxU32 Index1 = 0;	//x biggest axis
		PxU32 Index2 = 1;
		PxU32 Index3 = 2;
		if( (absNy > absPx) & (absNy > absNz))
		{
			//y biggest
			Index2 = 2;
			Index3 = 0;
			Index1 = 1;
		}
		else if(absNz > absPx)
		{
			//z biggest
			Index2 = 0;
			Index3 = 1;
			Index1 = 2;
		}

		const PxF32* data = &direction.x;
		const float Coeff = 1.0f / fabsf(data[Index1]);
		u = data[Index2] * Coeff;
		v = data[Index3] * Coeff;

		const PxU32 Sign = binary[Index1]>>31;
		return CubeIndex(Sign|(Index1+Index1));
	}

}

// ===== 원본: physx/source/geomutils/src/convex/GuShapeConvex.h
namespace px
{
namespace Gu
{
	struct PolygonalData;
	typedef void	(*HullPrefetchCB)		(PxU32 numVerts, const PxVec3*  verts);
	typedef void	(*HullProjectionCB)		(const PolygonalData& data, const PxVec3& dir, const PxMat34& world2hull, const Cm::FastVertex2ShapeScaling& scaling, PxReal& minimum, PxReal& maximum);
	typedef PxU32	(*SelectClosestEdgeCB)	(const PolygonalData& data, const Cm::FastVertex2ShapeScaling& scaling, const PxVec3& localDirection);

	struct PolygonalData
	{
		// Data
		Gu::InternalObjectsData				mInternal;
		PxMeshScale							mScale;
		PxVec3								mCenter;
		PxU32								mNbVerts;
		PxU32								mNbPolygons;
		PxU32								mNbEdges;
		const Gu::HullPolygonData*			mPolygons;
		const PxVec3*						mVerts;
		const PxU8*							mPolygonVertexRefs;
		const PxU8*							mFacesByEdges;
		const PxU16*						mVerticesByEdges;
		
		union
		{
			const Gu::BigConvexRawData*		mBigData;	// Only for big convexes
			const PxVec3*					mHalfSide;	// Only for boxes
		};

		// Code
		HullProjectionCB					mProjectHull;
		SelectClosestEdgeCB					mSelectClosestEdgeCB;

		EHD const PxU8*	getPolygonVertexRefs(const Gu::HullPolygonData& poly)	const
		{
			return mPolygonVertexRefs + poly.mVRef8;
		}
	};

	class PolygonalBox
	{
	public:
									EHDV PolygonalBox(const PxVec3& halfSide);

			EHDV void					getPolygonalData(PolygonalData*  dst)	const;

			const PxVec3&			mHalfSide;
			PxVec3					mVertices[8];
			Gu::HullPolygonData		mPolygons[6];
	private:
			PolygonalBox& operator=(const PolygonalBox&);
	};

	EHDV void getPolygonalData_Convex(PolygonalData*  dst, const Gu::ConvexHullData*  src, const Cm::FastVertex2ShapeScaling& scaling);
}
}

// ===== 원본: physx/source/geomutils/src/gjk/GuVecConvex.h
#define EPX_SUPPORT_INLINE EHD
#define EPX_SUPPORT_FORCE_INLINE EHD

namespace px
{
namespace Gu
{

	struct ConvexType
	{
		enum Type
		{
			eCONVEXHULL,
			eCONVEXHULLNOSCALE,
			eSPHERE,
			eBOX,
			eCAPSULE,
			eTRIANGLE,
			eTETRAHEDRON,
			eCUSTOM
		};
	};
	
	class ConvexV
	{
	public:

		EHD ConvexV(const ConvexType::Type type_) : type(type_), bMarginIsRadius(false)
		{
			margin = 0.f;
			minMargin = 0.f;
			sweepMargin = 0.f;
			center = aos::V3Zero();
		}

		EHD ConvexV(const ConvexType::Type type_, const aos::Vec3VArg center_) : type(type_), bMarginIsRadius(false)
		{
			using namespace aos;
			center = center_;
			margin = 0.f;
			minMargin = 0.f;
			sweepMargin = 0.f;
		}

		//everytime when someone transform the object, they need to up
		EHD void setCenter(const aos::Vec3VArg _center)
		{
			center = _center;
		}

		EHD void setMargin(const aos::FloatVArg margin_)
		{
			aos::FStore(margin_, &margin);
		}

		EHD void setMargin(const PxReal margin_)
		{
			margin = margin_;
		}


		EHD void setMinMargin(const aos::FloatVArg minMargin_)
		{
			aos::FStore(minMargin_, & minMargin);
		}

		EHD void setSweepMargin(const aos::FloatVArg sweepMargin_)
		{
			aos::FStore(sweepMargin_, &sweepMargin);
		}

		EHD aos::Vec3V getCenter()const 
		{
			return center;
		}

		EHD aos::FloatV getMargin() const
		{
			return aos::FLoad(margin);
		}

		EHD aos::FloatV getMinMargin() const
		{
			return aos::FLoad(minMargin);
		}

		EHD aos::FloatV getSweepMargin() const
		{
			return aos::FLoad(sweepMargin);
		}

		EHD ConvexType::Type getType() const
		{
			return type;
		}

		EHD aos::BoolV isMarginEqRadius()const
		{
			return aos::BLoad(bMarginIsRadius);
		}

		EHD bool getMarginIsRadius() const
		{
			return bMarginIsRadius;
		}

		EHD PxReal getMarginF() const
		{
			return margin;
		}


	protected:
		EHDI ~ConvexV(){}
		aos::Vec3V center;
		PxReal margin;				//margin is the amount by which we shrunk the shape for a convex or box. If the shape are sphere/capsule, margin is the radius
		PxReal minMargin;			//minMargin is some percentage of marginBase, which is used to determine the termination condition for gjk
		PxReal sweepMargin;			//sweepMargin minMargin is some percentage of marginBase, which is used to determine the termination condition for gjkRaycast
		ConvexType::Type	type;
		bool bMarginIsRadius;
	};

	EHD aos::FloatV getContactEps(const aos::FloatV& _marginA, const aos::FloatV& _marginB)
	{
		using namespace aos;

		const FloatV ratio = FLoad(0.25f);
		const FloatV minMargin = FMin(_marginA, _marginB);
		
		return FMul(minMargin, ratio);
	}

	EHD aos::FloatV getSweepContactEps(const aos::FloatV& _marginA, const aos::FloatV& _marginB)
	{
		using namespace aos;

		const FloatV ratio = FLoad(100.f);
		const FloatV minMargin = FAdd(_marginA, _marginB);
		
		return FMul(minMargin, ratio);
	}
}

}

// ===== 원본: physx/source/geomutils/src/convex/GuConvexSupportTable.h
namespace px
{
namespace Gu  
{

	class TriangleV; 
	class CapsuleV;
	class BoxV;
	class ConvexHullV;
	class ConvexHullNoScaleV;

	class SupportLocal
	{
	public:		
		aos::Vec3V shapeSpaceCenterOfMass;
		const aos::PxTransformV& transform;
		const aos::Mat33V& vertex2Shape;
		const aos::Mat33V& shape2Vertex;
		const bool isIdentityScale;	

		EHDI SupportLocal(const aos::PxTransformV& _transform, const aos::Mat33V& _vertex2Shape, const aos::Mat33V& _shape2Vertex, const bool _isIdentityScale = true): transform(_transform), 
			vertex2Shape(_vertex2Shape), shape2Vertex(_shape2Vertex), isIdentityScale(_isIdentityScale)
		{
		}

		EHD void setShapeSpaceCenterofMass(const aos::Vec3VArg _shapeSpaceCenterOfMass)
		{
			shapeSpaceCenterOfMass = _shapeSpaceCenterOfMass;
		}
        EHDI virtual ~SupportLocal() {}
		EHDV virtual aos::Vec3V doSupport(const aos::Vec3VArg dir) const = 0;
		EHDV virtual void doSupport(const aos::Vec3VArg dir, aos::FloatV& min, aos::FloatV& max) const = 0;
		EHDV virtual void populateVerts(const PxU8* inds, PxU32 numInds, const PxVec3* originalVerts, aos::Vec3V* verts)const = 0;
	
	protected:
		SupportLocal& operator=(const SupportLocal&);
	};

	template <typename Convex>
	class SupportLocalImpl : public SupportLocal
	{
		
	public:
		const Convex& conv;
		EHDI SupportLocalImpl(const Convex& _conv, const aos::PxTransformV& _transform, const aos::Mat33V& _vertex2Shape, const aos::Mat33V& _shape2Vertex, const bool _isIdentityScale = true) : 
		SupportLocal(_transform, _vertex2Shape, _shape2Vertex, _isIdentityScale), conv(_conv)
		{
		}

		EHDI aos::Vec3V doSupport(const aos::Vec3VArg dir) const
		{
			//return conv.supportVertsLocal(dir);
			return conv.supportLocal(dir);
		}

		EHDI void doSupport(const aos::Vec3VArg dir, aos::FloatV& min, aos::FloatV& max) const
		{
			return conv.supportLocal(dir, min, max);
		}

		EHDI void populateVerts(const PxU8* inds, PxU32 numInds, const PxVec3* originalVerts, aos::Vec3V* verts) const 
		{
			conv.populateVerts(inds, numInds, originalVerts, verts);
		}

	protected:
		SupportLocalImpl& operator=(const SupportLocalImpl&);

	};
}

}

// ===== 원본: physx/source/geomutils/src/gjk/GuVecConvexHull.h
namespace px
{
namespace Gu
{
#define CONVEX_MARGIN_RATIO			0.1f
#define CONVEX_MIN_MARGIN_RATIO		0.05f
#define	CONVEX_SWEEP_MARGIN_RATIO	0.025f
#define TOLERANCE_MARGIN_RATIO		0.08f
#define TOLERANCE_MIN_MARGIN_RATIO	0.05f

	//This margin is used in Persistent contact manifold
	EPX_SUPPORT_FORCE_INLINE aos::FloatV CalculatePCMConvexMargin(const Gu::ConvexHullData* hullData, const aos::Vec3VArg scale, 
		const PxReal toleranceLength, const PxReal toleranceRatio = TOLERANCE_MIN_MARGIN_RATIO)
	{
		
		using namespace aos;
		const Vec3V extents = V3Mul(V3LoadU_SafeReadW(hullData->mInternal.mInternalExtents), scale);
		const FloatV min = V3ExtractMin(extents);
		const FloatV toleranceMargin = FLoad(toleranceLength * toleranceRatio);
		//ML: 25% of the minimum extents of the internal AABB as this convex hull's margin
		return FMin(FMul(min, FLoad(0.25f)), toleranceMargin);
	}

	EPX_SUPPORT_FORCE_INLINE aos::FloatV CalculateMTDConvexMargin(const Gu::ConvexHullData* hullData, const aos::Vec3VArg scale)
	{
		using namespace aos;
		const Vec3V extents = V3Mul(V3LoadU_SafeReadW(hullData->mInternal.mInternalExtents), scale);
		const FloatV min = V3ExtractMin(extents);
		//ML: 25% of the minimum extents of the internal AABB as this convex hull's margin
		return FMul(min, FLoad(0.25f));
	}

	//This minMargin is used in PCM contact gen
	EPX_SUPPORT_FORCE_INLINE void CalculateConvexMargin(const InternalObjectsData& internalObject, PxReal& margin, PxReal& minMargin, PxReal& sweepMargin,
		const aos::Vec3VArg scale)
	{
		using namespace aos;
		
		const Vec3V extents = V3Mul(V3LoadU_SafeReadW(internalObject.mInternalExtents), scale);
		const FloatV min_ = V3ExtractMin(extents);

		PxReal minExtent;
		FStore(min_, &minExtent);

		//margin is used as acceptanceTolerance for overlap.
		margin = minExtent * CONVEX_MARGIN_RATIO;
		//minMargin is used in the GJK termination condition
		minMargin = minExtent * CONVEX_MIN_MARGIN_RATIO;
		//this is used for sweep(gjkRaycast)
		sweepMargin = minExtent * CONVEX_SWEEP_MARGIN_RATIO;
	}


	EPX_SUPPORT_FORCE_INLINE aos::Mat33V ConstructSkewMatrix(const aos::Vec3VArg scale, const aos::QuatVArg rotation) 
	{
		using namespace aos;
		Mat33V rot;
		QuatGetMat33V(rotation, rot.col0, rot.col1, rot.col2);
		Mat33V trans = M33Trnsps(rot);
		trans.col0 = V3Scale(trans.col0, V3GetX(scale));
		trans.col1 = V3Scale(trans.col1, V3GetY(scale));
		trans.col2 = V3Scale(trans.col2, V3GetZ(scale));
		return M33MulM33(trans, rot);
	}

	EPX_SUPPORT_FORCE_INLINE void ConstructSkewMatrix(const aos::Vec3VArg scale, const aos::QuatVArg rotation, aos::Mat33V& vertex2Shape, aos::Mat33V& shape2Vertex, aos::Vec3V& center, const bool idtScale) 
	{
		using namespace aos;

		E_ASSERT(!V3AllEq(scale, V3Zero()));
	
		if(idtScale)
		{
			//create identity buffer
			const Mat33V identity = M33Identity();
			vertex2Shape = identity;
			shape2Vertex = identity;
		}
		else
		{
			const FloatV scaleX = V3GetX(scale);
			const Vec3V invScale = V3Recip(scale);

			//this is uniform scale
			if(V3AllEq(V3Splat(scaleX), scale))
			{	
				vertex2Shape = M33Diagonal(scale);
				shape2Vertex = M33Diagonal(invScale);
			}
			else
			{
				Mat33V rot;
				QuatGetMat33V(rotation, rot.col0, rot.col1, rot.col2);
				const Mat33V trans = M33Trnsps(rot);
				/*
					vertex2shape
					skewMat = Inv(R)*Diagonal(scale)*R;
				*/

				const Mat33V temp(V3Scale(trans.col0, scaleX), V3Scale(trans.col1, V3GetY(scale)), V3Scale(trans.col2, V3GetZ(scale)));
				vertex2Shape = M33MulM33(temp, rot);

				//don't need it in the support function
				/*
					shape2Vertex
					invSkewMat =(invSkewMat)= Inv(R)*Diagonal(1/scale)*R;
				*/
				
				shape2Vertex.col0 = V3Scale(trans.col0, V3GetX(invScale));
				shape2Vertex.col1 = V3Scale(trans.col1, V3GetY(invScale));
				shape2Vertex.col2 = V3Scale(trans.col2, V3GetZ(invScale));
				shape2Vertex = M33MulM33(shape2Vertex, rot);

				//shape2Vertex = M33Inverse(vertex2Shape);
			}

			//transform center to shape space
			center = M33MulV3(vertex2Shape, center);
		}
	}

	EPX_SUPPORT_FORCE_INLINE aos::Mat33V ConstructVertex2ShapeMatrix(const aos::Vec3VArg scale, const aos::QuatVArg rotation) 
	{
		using namespace aos;
		Mat33V rot;
		QuatGetMat33V(rotation, rot.col0, rot.col1, rot.col2);
		const Mat33V trans = M33Trnsps(rot);
		/*
			vertex2shape
			skewMat = Inv(R)*Diagonal(scale)*R;
		*/

		const Mat33V temp(V3Scale(trans.col0, V3GetX(scale)), V3Scale(trans.col1, V3GetY(scale)), V3Scale(trans.col2, V3GetZ(scale)));
		return M33MulM33(temp, rot);
	}


	class ConvexHullV : public ConvexV
	{

		class TinyBitMap
		{
		public:
			PxU32 m[8];
			EHD TinyBitMap() { m[0] = m[1] = m[2] = m[3] = m[4] = m[5] = m[6] = m[7] = 0; }
			EHD void set(PxU8 v) { m[v >> 5] |= 1 << (v & 31); }
			EHD bool get(PxU8 v) const { return (m[v >> 5] & 1 << (v & 31)) != 0; }
		};


	public:
		/**
		\brief Constructor
		*/
		EPX_SUPPORT_INLINE ConvexHullV() : ConvexV(ConvexType::eCONVEXHULL)
		{
		}

		EPX_SUPPORT_INLINE ConvexHullV(const Gu::ConvexHullData* _hullData, const aos::Vec3VArg _center, const aos::Vec3VArg scale, const aos::QuatVArg scaleRot,
			const bool idtScale) :
			ConvexV(ConvexType::eCONVEXHULL, _center)
		{
			using namespace aos;

			hullData = _hullData;
			const PxVec3*  tempVerts = _hullData->getHullVertices();
			verts = tempVerts;
			numVerts = _hullData->mNbHullVertices;
			CalculateConvexMargin(_hullData->mInternal, margin, minMargin, sweepMargin, scale);
			ConstructSkewMatrix(scale, scaleRot, vertex2Shape, shape2Vertex, center, idtScale);
			data = _hullData->mBigConvexRawData;
		}

		EPX_SUPPORT_INLINE ConvexHullV(const Gu::ConvexHullData* _hullData, const aos::Vec3VArg _center) :
			ConvexV(ConvexType::eCONVEXHULL, _center)
		{
			using namespace aos;

			hullData = _hullData;
			verts = _hullData->getHullVertices();
			numVerts = _hullData->mNbHullVertices;
			data = _hullData->mBigConvexRawData;
		}

		//this is used by CCD system
		EPX_SUPPORT_INLINE ConvexHullV(const PxGeometry& geom) : ConvexV(ConvexType::eCONVEXHULL, aos::V3Zero())
		{
			using namespace aos;
			const PxConvexMeshGeometry& convexGeom = static_cast<const PxConvexMeshGeometry&>(geom);
			const Gu::ConvexHullData* hData = _getHullData(convexGeom);

			const Vec3V vScale = V3LoadU_SafeReadW(convexGeom.scale.scale);	// PT: safe because 'rotation' follows 'scale' in PxMeshScale
			const QuatV vRot = QuatVLoadU(&convexGeom.scale.rotation.x);
			const bool idtScale = convexGeom.scale.isIdentity();

			hullData = hData;
			const PxVec3*  tempVerts = hData->getHullVertices();
			verts = tempVerts;
			numVerts = hData->mNbHullVertices;
			CalculateConvexMargin(hData->mInternal, margin, minMargin, sweepMargin, vScale);
			ConstructSkewMatrix(vScale, vRot, vertex2Shape, shape2Vertex, center, idtScale);

			data = hData->mBigConvexRawData;
		}

		//this is used by convex vs tetrahedron collision
		EPX_SUPPORT_INLINE ConvexHullV(const Gu::PolygonalData& polyData, const Cm::FastVertex2ShapeScaling& convexScale) :
			ConvexV(ConvexType::eCONVEXHULL, aos::V3LoadU(polyData.mCenter))
		{
			using namespace aos;

			const Vec3V vScale = V3LoadU(polyData.mScale.scale);

			verts = polyData.mVerts;
			numVerts = PxU8(polyData.mNbVerts);
			CalculateConvexMargin(polyData.mInternal, margin, minMargin, sweepMargin, vScale);

			const PxMat33& v2s = convexScale.getVertex2ShapeSkew();
			const PxMat33& s2v = convexScale.getShape2VertexSkew();

			vertex2Shape.col0 = V3LoadU(v2s.column0);
			vertex2Shape.col1 = V3LoadU(v2s.column1);
			vertex2Shape.col2 = V3LoadU(v2s.column2);

			shape2Vertex.col0 = V3LoadU(s2v.column0);
			shape2Vertex.col1 = V3LoadU(s2v.column1);
			shape2Vertex.col2 = V3LoadU(s2v.column2);

			data = polyData.mBigData;

		}

		EPX_SUPPORT_INLINE void initialize(const Gu::ConvexHullData* _hullData, const aos::Vec3VArg _center, const aos::Vec3VArg scale,
			const aos::QuatVArg scaleRot, const bool idtScale)
		{
			using namespace aos;

			const PxVec3* tempVerts = _hullData->getHullVertices();
			CalculateConvexMargin(_hullData->mInternal, margin, minMargin, sweepMargin, scale);
			ConstructSkewMatrix(scale, scaleRot, vertex2Shape, shape2Vertex, center, idtScale);

			verts = tempVerts;
			numVerts = _hullData->mNbHullVertices;
			//rot = _rot;	

			center = _center;

			//	searchIndex = 0;
			data = _hullData->mBigConvexRawData;

			hullData = _hullData;
			if (_hullData->mBigConvexRawData)
			{
				PxPrefetchLine(hullData->mBigConvexRawData->mValencies);
				PxPrefetchLine(hullData->mBigConvexRawData->mValencies, 128);
				PxPrefetchLine(hullData->mBigConvexRawData->mAdjacentVerts);
			}
		}


	
		EHD void resetMargin(const PxReal toleranceLength)
		{
			const PxReal toleranceMinMargin = toleranceLength * TOLERANCE_MIN_MARGIN_RATIO;
			const PxReal toleranceMargin = toleranceLength * TOLERANCE_MARGIN_RATIO;

			margin = PxMin(margin, toleranceMargin);
			minMargin = PxMin(minMargin, toleranceMinMargin);
		}

		EHD aos::Vec3V supportPoint(const PxI32 index)const
		{
			using namespace aos;
			
			return M33MulV3(vertex2Shape, V3LoadU_SafeReadW(verts[index]));	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
		}

		 EHDI PxU32 hillClimbing(const aos::Vec3VArg _dir)const
		{
			using namespace aos;

			const Gu::Valency* valency = data->mValencies;
			const PxU8* adjacentVerts = data->mAdjacentVerts;
			
			//NotSoTinyBitMap visited;
			PxU32 smallBitMap[8] = {0,0,0,0,0,0,0,0};

		//	PxU32 index = searchIndex;
			PxU32 index = 0;

			{
				PxVec3 vertexSpaceDirection;
				V3StoreU(_dir, vertexSpaceDirection);
				const PxU32 offset = ComputeCubemapNearestOffset(vertexSpaceDirection, data->mSubdiv);
				//const PxU32 offset = ComputeCubemapOffset(vertexSpaceDirection, data->mSubdiv);
				index = data->mSamples[offset];
			}

			Vec3V maxPoint = V3LoadU_SafeReadW(verts[index]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
			FloatV max = V3Dot(maxPoint, _dir);
	
			PxU32 initialIndex = index;
			
			do
			{
				initialIndex = index;
				const PxU32 numNeighbours = valency[index].mCount;
				const PxU32 offset = valency[index].mOffset;

				for(PxU32 a = 0; a < numNeighbours; ++a)
				{
					const PxU32 neighbourIndex = adjacentVerts[offset + a];

					const Vec3V vertex = V3LoadU_SafeReadW(verts[neighbourIndex]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
					const FloatV dist = V3Dot(vertex, _dir);
					if(FAllGrtr(dist, max))
					{
						const PxU32 ind = neighbourIndex>>5;
						const PxU32 mask = PxU32(1 << (neighbourIndex & 31));
						if((smallBitMap[ind] & mask) == 0)
						{
							smallBitMap[ind] |= mask;
							max = dist;
							index = neighbourIndex;
						}
					}
				}

			}while(index != initialIndex);

			return index;
		}

		EPX_SUPPORT_INLINE PxU32 bruteForceSearch(const aos::Vec3VArg _dir)const 
		{
			using namespace aos;
			//brute force
			PxVec3 dir;
			V3StoreU(_dir, dir);

			PxReal max = verts[0].dot(dir);
			PxU32 maxIndex = 0;

			for (PxU32 i = 1; i < numVerts; ++i)
			{
				const PxReal dist = verts[i].dot(dir);
				if (dist > max)
				{
					max = dist;
					maxIndex = i;
				}
			}
			return maxIndex;
		}

		//points are in vertex space, _dir in vertex space
		 EHDI PxU32 supportVertexIndex(const aos::Vec3VArg _dir)const
		{
			using namespace aos;
			if(data)
				return hillClimbing(_dir);
			else
				return bruteForceSearch(_dir);
		}

		//dir is in the vertex space
		EPX_SUPPORT_INLINE void bruteForceSearchMinMax(const aos::Vec3VArg _dir, aos::FloatV& min, aos::FloatV& max)const 
		{
			using namespace aos;

			//brute force
			PxVec3 dir;
			V3StoreU(_dir, dir);
			//get the support point from the orignal margin
			PxReal _max = verts[0].dot(dir);
			PxReal _min = _max;

			for(PxU32 i = 1; i < numVerts; ++i)
			{ 
				const PxReal dist = verts[i].dot(dir);
				_max = PxMax(dist, _max);
				_min = PxMin(dist, _min);
			}
			min = FLoad(_min);
			max = FLoad(_max);
		}

		//This function is used in the full contact manifold generation code, points are in vertex space.
		//This function support scaling, _dir is in the shape space	
		EPX_SUPPORT_INLINE void supportVertexMinMax(const aos::Vec3VArg _dir, aos::FloatV& min, aos::FloatV& max)const
		{
			using namespace aos;

			//dir is in the vertex space
			const Vec3V dir = M33TrnspsMulV3(vertex2Shape, _dir);

			if(data)
			{
				const PxU32 maxIndex = hillClimbing(dir);
				const PxU32 minIndex = hillClimbing(V3Neg(dir));
				const Vec3V maxPoint = M33MulV3(vertex2Shape, V3LoadU_SafeReadW(verts[maxIndex]));	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
				const Vec3V minPoint = M33MulV3(vertex2Shape, V3LoadU_SafeReadW(verts[minIndex]));	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
				min = V3Dot(_dir, minPoint);
				max = V3Dot(_dir, maxPoint);
			}
			else
			{
				//dir is in the vertex space
				bruteForceSearchMinMax(dir, min, max);
			}
		}  
 
		//This function is used in the full contact manifold generation code
		EPX_SUPPORT_INLINE void populateVerts(const PxU8* inds, PxU32 numInds, const PxVec3* originalVerts, aos::Vec3V* _verts)const
		{
			using namespace aos;

			for(PxU32 i=0; i<numInds; ++i)
				_verts[i] = M33MulV3(vertex2Shape, V3LoadU_SafeReadW(originalVerts[inds[i]]));	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'populateVerts' is always called with polyData.mVerts)
		}

		//This function is used in epa
		//dir is in the shape space
		EPX_SUPPORT_INLINE aos::Vec3V supportLocal(const aos::Vec3VArg dir)const
		{
			using namespace aos;
			//scale dir and put it in the vertex space
			const Vec3V _dir = M33TrnspsMulV3(vertex2Shape, dir);
			const PxU32 maxIndex = supportVertexIndex(_dir);
			return M33MulV3(vertex2Shape, V3LoadU_SafeReadW(verts[maxIndex]));	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
		}

		//this is used in the sat test for the full contact gen
		EPX_SUPPORT_INLINE void supportLocal(const aos::Vec3VArg dir, aos::FloatV& min, aos::FloatV& max)const
		{
			using namespace aos;
			//dir is in the shape space
			supportVertexMinMax(dir, min, max);
		}

		//This function is used in epa
		EPX_SUPPORT_INLINE aos::Vec3V supportRelative(const aos::Vec3VArg dir, const aos::PxMatTransformV& aTob, const aos::PxMatTransformV& aTobT) const
		{
			using namespace aos;
		
			//transform dir into the shape space
//			const Vec3V dir_ = aTob.rotateInv(dir);//relTra.rotateInv(dir);
			const Vec3V dir_ = aTobT.rotate(dir);//relTra.rotateInv(dir);
			const Vec3V maxPoint = supportLocal(dir_);
			//translate maxPoint from shape space of a back to the b space
			return aTob.transform(maxPoint);//relTra.transform(maxPoint);
		}

		//dir in the shape space, this function is used in gjk	
		EPX_SUPPORT_INLINE aos::Vec3V supportLocal(const aos::Vec3VArg dir, PxI32& index)const
		{
			using namespace aos;
			//scale dir and put it in the vertex space, for non-uniform scale, we don't want the scale in the dir, therefore, we are using
			//the transpose of the inverse of shape2Vertex(which is vertex2shape). This will allow us igore the scale and keep the rotation
			const Vec3V dir_ = M33TrnspsMulV3(vertex2Shape, dir);
			//get the extreme point index
			const PxU32 maxIndex = supportVertexIndex(dir_);
			index = PxI32(maxIndex);
			//p is in the shape space
			return M33MulV3(vertex2Shape, V3LoadU_SafeReadW(verts[index]));	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
		}

		//this function is used in gjk	
		EPX_SUPPORT_INLINE aos::Vec3V supportRelative(	const aos::Vec3VArg dir, const aos::PxMatTransformV& aTob,
															const aos::PxMatTransformV& aTobT, PxI32& index)const
		{
			using namespace aos;

			//transform dir from b space to the shape space of a space
//			const Vec3V dir_ = aTob.rotateInv(dir);//relTra.rotateInv(dir);//M33MulV3(skewInvRot, dir);
			const Vec3V dir_ = aTobT.rotate(dir);//relTra.rotateInv(dir);//M33MulV3(skewInvRot, dir);
			const Vec3V p = supportLocal(dir_, index);
			//transfrom from a to b space
			return aTob.transform(p);
		}

		aos::Mat33V vertex2Shape;//inv(R)*S*R
		aos::Mat33V shape2Vertex;//inv(vertex2Shape)

		const Gu::ConvexHullData* hullData;
		const BigConvexRawData* data;  
		const PxVec3* verts;
		PxU8 numVerts;
	};

}

}

// ===== 원본: physx/source/geomutils/src/gjk/GuVecConvexHullNoScale.h
namespace px
{
namespace Gu
{
	class ConvexHullNoScaleV : public ConvexHullV
	{
		public:
		/**
		\brief Constructor
		*/
		EPX_SUPPORT_INLINE ConvexHullNoScaleV(): ConvexHullV()
		{
		}

		EHD aos::Vec3V supportPoint(const PxI32 index)const
		{
			using namespace aos;
			return V3LoadU_SafeReadW(verts[index]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
		}

		//This funcation is just to load the PxVec3 to Vec3V. However, for GuVecConvexHul.h, this is used to transform all the verts from vertex space to shape space
		EPX_SUPPORT_INLINE void populateVerts(const PxU8* inds, PxU32 numInds, const PxVec3* originalVerts, aos::Vec3V* verts_)const
		{
			using namespace aos;

			for(PxU32 i=0; i<numInds; ++i)
				verts_[i] = V3LoadU_SafeReadW(originalVerts[inds[i]]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'populateVerts' is always called with polyData.mVerts)
		}
		
		//This function is used in epa
		//dir is in the shape space
		EPX_SUPPORT_INLINE aos::Vec3V supportLocal(const aos::Vec3VArg dir)const
		{
			using namespace aos;
			const PxU32 maxIndex = supportVertexIndex(dir);
			return V3LoadU_SafeReadW(verts[maxIndex]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
		}

		//this is used in the sat test for the full contact gen
		EPX_SUPPORT_INLINE void supportLocal(const aos::Vec3VArg dir, aos::FloatV& min, aos::FloatV& max)const
		{
			supportVertexMinMax(dir, min, max);
		}

		//This function is used in epa
		EPX_SUPPORT_INLINE aos::Vec3V supportRelative(const aos::Vec3VArg dir, const aos::PxMatTransformV& aTob, const aos::PxMatTransformV& aTobT) const
		{
			using namespace aos;
		
			//transform dir into the shape space
			const Vec3V _dir = aTobT.rotate(dir);//relTra.rotateInv(dir);
			const Vec3V maxPoint = supportLocal(_dir);
			//translate maxPoint from shape space of a back to the b space
			return aTob.transform(maxPoint);//relTra.transform(maxPoint);
		}

		//dir in the shape space, this function is used in gjk
		EPX_SUPPORT_INLINE aos::Vec3V supportLocal(const aos::Vec3VArg dir, PxI32& index)const
		{
			using namespace aos;
			//scale dir and put it in the vertex space, for non-uniform scale, we don't want the scale in the dir, therefore, we are using
			//the transpose of the inverse of shape2Vertex(which is vertex2shape). This will allow us igore the scale and keep the rotation
			//get the extreme point index
			const PxU32 maxIndex = supportVertexIndex(dir);
			index = PxI32(maxIndex);
			return V3LoadU_SafeReadW(verts[index]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
		}

		//this function is used in gjk
		EPX_SUPPORT_INLINE aos::Vec3V supportRelative(	const aos::Vec3VArg dir, const aos::PxMatTransformV& aTob,
														const aos::PxMatTransformV& aTobT, PxI32& index)const
		{
			using namespace aos;

			//transform dir from b space to the shape space of a space
			const Vec3V _dir = aTobT.rotate(dir);//relTra.rotateInv(dir);//M33MulV3(skewInvRot, dir);
			const Vec3V p = supportLocal(_dir, index);
			//transfrom from a to b space
			return aTob.transform(p);
		}

		EPX_SUPPORT_INLINE void bruteForceSearchMinMax(const aos::Vec3VArg _dir, aos::FloatV& min, aos::FloatV& max)const 
		{
			using namespace aos;
			//brute force
			//get the support point from the orignal margin
			FloatV _max = V3Dot(V3LoadU_SafeReadW(verts[0]), _dir);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
			FloatV _min = _max;

			for(PxU32 i = 1; i < numVerts; ++i)
			{ 
				PxPrefetchLine(&verts[i], 128);
				const Vec3V vertex = V3LoadU_SafeReadW(verts[i]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
				const FloatV dist = V3Dot(vertex, _dir);

				_max = FMax(dist, _max);
				_min = FMin(dist, _min);
			}

			min = _min;
			max = _max;
		}

		//This function support no scaling, dir is in the shape space(the same as vertex space)
		EPX_SUPPORT_INLINE void supportVertexMinMax(const aos::Vec3VArg dir, aos::FloatV& min, aos::FloatV& max)const
		{
			using namespace aos;

			if(data)
			{
				const PxU32 maxIndex = hillClimbing(dir);
				const PxU32 minIndex = hillClimbing(V3Neg(dir));
				const Vec3V maxPoint = V3LoadU_SafeReadW(verts[maxIndex]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
				const Vec3V minPoint = V3LoadU_SafeReadW(verts[minIndex]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'verts' is initialized with ConvexHullData::getHullVertices())
				min = V3Dot(dir, minPoint);
				max = V3Dot(dir, maxPoint);
			}
			else
			{
				bruteForceSearchMinMax(dir, min, max);
			}
		}  
	};

	#define EPX_CONVEX_TO_NOSCALECONVEX(x)	(static_cast<const ConvexHullNoScaleV*>(x))
}

}

// ===== 원본: physx/source/geomutils/src/gjk/GuVecBox.h
namespace px
{
 

namespace Gu
{

#define BOX_MARGIN_RATIO		0.15f
#define	BOX_MIN_MARGIN_RATIO	0.05f
#define	BOX_SWEEP_MARGIN_RATIO	0.05f

#define BOX_MARGIN_CCD_RATIO		0.01f
#define	BOX_MIN_MARGIN_CCD_RATIO	0.005f


	class CapsuleV;


	EHD void CalculateBoxMargin(const aos::Vec3VArg extent, PxReal& margin, PxReal& minMargin, PxReal& sweepMargin,
		const PxReal marginR = BOX_MARGIN_RATIO, const PxReal minMarginR = BOX_MIN_MARGIN_RATIO)
	{
		using namespace aos;

		PxReal minExtent;
		const FloatV min = V3ExtractMin(extent);
		FStore(min, &minExtent);

		margin = minExtent * marginR;
		minMargin = minExtent * minMarginR;
		sweepMargin = minExtent * BOX_SWEEP_MARGIN_RATIO;
	}

	EHD aos::FloatV CalculateBoxTolerance(const aos::Vec3VArg extent)
	{
		using namespace aos;
		
		const FloatV r0 = FLoad(0.01f);
		const FloatV min = V3ExtractMin(extent);//FMin(V3GetX(extent), FMin(V3GetY(extent), V3GetZ(extent)));
		return FMul(min, r0);
	}

	//This method is called in the PCM contact gen for the refreshing contacts 
	EHD aos::FloatV CalculatePCMBoxMargin(const aos::Vec3VArg extent, const PxReal toleranceLength, const PxReal toleranceMarginRatio = BOX_MARGIN_RATIO)
	{
		using namespace aos;
	
		const FloatV min = V3ExtractMin(extent);//FMin(V3GetX(extent), FMin(V3GetY(extent), V3GetZ(extent)));
		const FloatV toleranceMargin = FLoad(toleranceLength * toleranceMarginRatio);
		return FMin(FMul(min, FLoad(BOX_MARGIN_RATIO)), toleranceMargin);
	}

	EHD aos::FloatV CalculateMTDBoxMargin(const aos::Vec3VArg extent)
	{
		using namespace aos;

		const FloatV min = V3ExtractMin(extent);//FMin(V3GetX(extent), FMin(V3GetY(extent), V3GetZ(extent)));
		return FMul(min, FLoad(BOX_MARGIN_RATIO));
	}

	class BoxV : public ConvexV
	{
	public:

		/**
		\brief Constructor
		*/
		EHDI BoxV() : ConvexV(ConvexType::eBOX)
		{
		}

		EHD BoxV(const aos::Vec3VArg origin, const aos::Vec3VArg extent) : 
			ConvexV(ConvexType::eBOX, origin), extents(extent)
		{
			CalculateBoxMargin(extent, margin, minMargin, sweepMargin);
		}
		
		//this constructor is used by the CCD system
		EHD BoxV(const PxGeometry& geom) : ConvexV(ConvexType::eBOX, aos::V3Zero())
		{
			using namespace aos;
			const PxBoxGeometry& boxGeom = static_cast<const PxBoxGeometry&>(geom);
			const Vec3V extent = aos::V3LoadU(boxGeom.halfExtents);
			extents = extent;
			CalculateBoxMargin(extent, margin, minMargin, sweepMargin, BOX_MARGIN_CCD_RATIO, BOX_MIN_MARGIN_CCD_RATIO);
		}

		/**
		\brief Destructor
		*/
		EHDI ~BoxV()
		{
		}

		EHD void resetMargin(const PxReal toleranceLength)
		{
			minMargin = PxMin(toleranceLength * BOX_MIN_MARGIN_RATIO, minMargin);
		}

		//! Assignment operator
		EHD const BoxV& operator=(const BoxV& other)
		{
			center	= other.center;
			extents	= other.extents;
			margin =  other.margin;
			minMargin = other.minMargin;
			sweepMargin = other.sweepMargin;
			return *this;
		}

		EHD void populateVerts(const PxU8* inds, PxU32 numInds, const PxVec3* originalVerts, aos::Vec3V* verts)const
		{
			using namespace aos;

			for(PxU32 i=0; i<numInds; ++i)
				verts[i] = V3LoadU_SafeReadW(originalVerts[inds[i]]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData (and 'populateVerts' is always called with polyData.mVerts)
		}

		EHD aos::Vec3V supportPoint(const PxI32 index)const
		{
			using namespace aos;
			const BoolV con = boxVertexTable_get(index);
			return V3Sel(con, extents, V3Neg(extents));
		}  

		EHD void getIndex(const aos::BoolV con, PxI32& index)const 
		{ 
			using namespace aos;
			index = PxI32(BGetBitMask(con) & 0x7); 
		}

		EHD aos::Vec3V supportLocal(const aos::Vec3VArg dir)const  
		{
			using namespace aos;
			return V3Sel(V3IsGrtr(dir, V3Zero()), extents, V3Neg(extents));
		}

		//this is used in the sat test for the full contact gen
		EPX_SUPPORT_INLINE void supportLocal(const aos::Vec3VArg dir, aos::FloatV& min, aos::FloatV& max)const
		{
			using namespace aos;
			const Vec3V point = V3Sel(V3IsGrtr(dir, V3Zero()), extents, V3Neg(extents));
			max = V3Dot(dir, point);
			min = FNeg(max);
		}

		EPX_SUPPORT_INLINE aos::Vec3V supportRelative(const aos::Vec3VArg dir, const aos::PxMatTransformV& aTob, const aos::PxMatTransformV& aTobT) const  
		{
			//a is the current object, b is the other object, dir is in the local space of b
			using namespace aos;
//			const Vec3V _dir = aTob.rotateInv(dir);//relTra.rotateInv(dir);//from b to a
			const Vec3V _dir = aTobT.rotate(dir);//relTra.rotateInv(dir);//from b to a
			const Vec3V p =  supportLocal(_dir);
			//transfer p into the b space
			return aTob.transform(p);//relTra.transform(p);
		}

		EPX_SUPPORT_INLINE aos::Vec3V supportLocal(const aos::Vec3VArg dir, PxI32& index)const  
		{
			using namespace aos;
			const BoolV comp = V3IsGrtr(dir, V3Zero());
			getIndex(comp, index);
			return  V3Sel(comp, extents, V3Neg(extents));
		}

		EPX_SUPPORT_INLINE aos::Vec3V supportRelative(	const aos::Vec3VArg dir, const aos::PxMatTransformV& aTob,
															const aos::PxMatTransformV& aTobT, PxI32& index)const  
		{
			//a is the current object, b is the other object, dir is in the local space of b
			using namespace aos;
//			const Vec3V _dir = aTob.rotateInv(dir);//relTra.rotateInv(dir);//from b to a
			const Vec3V _dir = aTobT.rotate(dir);//relTra.rotateInv(dir);//from b to a
			const Vec3V p = supportLocal(_dir, index);
			//transfer p into the b space
			return aTob.transform(p);//relTra.transform(p);
		}

		aos::Vec3V  extents;
	};
}	//E_STATIC_ASSERT(sizeof(Gu::BoxV) == 96);

}

// ===== 원본: physx/source/geomutils/src/gjk/GuVecCapsule.h
namespace px
{
namespace Gu
{

	EHD aos::FloatV CalculateCapsuleMinMargin(const aos::FloatVArg radius)
	{
		using namespace aos;
		const FloatV ratio = aos::FLoad(0.05f);
		return FMul(radius, ratio);
	}

	class CapsuleV : public ConvexV  
	{
	public:
		/**
		\brief Constructor
		*/

		EHDI CapsuleV():ConvexV(ConvexType::eCAPSULE)
		{
			bMarginIsRadius = true;
		}

		//constructor for sphere
		EHDI CapsuleV(const aos::Vec3VArg p, const aos::FloatVArg radius_) : ConvexV(ConvexType::eCAPSULE)
		{
			using namespace aos;
			center = p;
			radius = radius_;
			p0 = p;
			p1 = p;
			FStore(radius, &margin);
			FStore(radius, &minMargin);
			FStore(radius, &sweepMargin);
			bMarginIsRadius = true; 
		}

		EHDI CapsuleV(const aos::Vec3VArg center_, const aos::Vec3VArg v_, const aos::FloatVArg radius_) : 
			ConvexV(ConvexType::eCAPSULE, center_)
		{
			using namespace aos;
			radius = radius_;
			p0 = V3Add(center_, v_);
			p1 = V3Sub(center_, v_);
			FStore(radius, &margin);
			FStore(radius, &minMargin);
			FStore(radius, &sweepMargin);
			bMarginIsRadius = true;
		}

		EHDI CapsuleV(const PxGeometry& geom) : ConvexV(ConvexType::eCAPSULE, aos::V3Zero())
		{
			using namespace aos;
			const PxCapsuleGeometry& capsuleGeom = static_cast<const PxCapsuleGeometry&>(geom);

			const Vec3V axis = V3Scale(V3UnitX(), FLoad(capsuleGeom.halfHeight));
			const FloatV r = FLoad(capsuleGeom.radius);
			p0 = axis;
			p1 = V3Neg(axis);
			radius = r;
			FStore(radius, &margin);
			FStore(radius, &minMargin);
			FStore(radius, &sweepMargin);
			bMarginIsRadius = true;
		}

		/**
		\brief Constructor

		\param _radius Radius of the capsule.
		*/

		/**
		\brief Destructor
		*/
		EHDI ~CapsuleV()
		{
		}

		EHD void initialize(const aos::Vec3VArg _p0, const aos::Vec3VArg _p1, const aos::FloatVArg _radius)
		{
			using namespace aos;
			radius = _radius;
			p0 = _p0;
			p1 = _p1;
			FStore(radius, &margin);
			FStore(radius, &minMargin);
			FStore(radius, &sweepMargin);
			center = V3Scale(V3Add(_p0, _p1), FHalf());
		}   

		EHDI aos::Vec3V computeDirection() const
		{
			return aos::V3Sub(p1, p0);
		}

		EHD	aos::FloatV	getRadius()	const
		{
			return radius;
		}

		EHD aos::Vec3V supportPoint(const PxI32 index)const
		{
			return (&p0)[1-index];
		}

		EHD void getIndex(const aos::BoolV con, PxI32& index)const
		{
			using namespace aos;
			const VecI32V v = VecI32V_From_BoolV(con);
			const VecI32V t = VecI32V_And(v, VecI32V_One());
			PxI32_From_VecI32V(t, &index);
		}

		EHD void setCenter(const aos::Vec3VArg _center)
		{
			using namespace aos;
			Vec3V offset = V3Sub(_center, center);
			center = _center;

			p0 = V3Add(p0, offset);
			p1 = V3Add(p1, offset);
		}

		//dir, p0 and p1 are in the local space of dir
		EHD aos::Vec3V supportLocal(const aos::Vec3VArg dir)const
		{
			using namespace aos;
			//const Vec3V _dir = V3Normalize(dir);
			const FloatV dist0 = V3Dot(p0, dir);
			const FloatV dist1 = V3Dot(p1, dir);
			return V3Sel(FIsGrtr(dist0, dist1), p0, p1);
		}
	
		EHD aos::Vec3V supportRelative(const aos::Vec3VArg dir, const aos::PxMatTransformV& aToB, const aos::PxMatTransformV& aTobT) const
		{
			using namespace aos;
			//transform dir into the local space of a
//			const Vec3V _dir = aToB.rotateInv(dir);
			const Vec3V _dir = aTobT.rotate(dir);
			const Vec3V p = supportLocal(_dir);
			//transform p back to the local space of b
			return aToB.transform(p);
		}

		//dir, p0 and p1 are in the local space of dir
		EHD aos::Vec3V supportLocal(const aos::Vec3VArg dir, PxI32& index)const
		{
			using namespace aos;
			
			const FloatV dist0 = V3Dot(p0, dir);
			const FloatV dist1 = V3Dot(p1, dir);
			const BoolV comp = FIsGrtr(dist0, dist1);
			getIndex(comp, index);
			return V3Sel(comp, p0, p1);
		}
	
		EHD aos::Vec3V supportRelative(	const aos::Vec3VArg dir, const aos::PxMatTransformV& aToB,
														const aos::PxMatTransformV& aTobT, PxI32& index)const
		{
			using namespace aos;
			//transform dir into the local space of a
//			const Vec3V _dir = aToB.rotateInv(dir);
			const Vec3V _dir = aTobT.rotate(dir);

			const Vec3V p = supportLocal(_dir, index);
			//transform p back to the local space of b
			return aToB.transform(p);
		}


		EHD aos::Vec3V supportLocal(aos::Vec3V& support, const PxI32& index, const aos::BoolV comp)const
		{
			E_UNUSED(index);

			using namespace aos;
			const Vec3V p = V3Sel(comp, p0, p1);
			support = p;
			return p;
		}

		EHD aos::FloatV getSweepMargin() const
		{
			return aos::FZero();
		}

		//don't change the order of p0 and p1, the getPoint function depend on the order
		aos::Vec3V	p0;		//!< Start of segment
		aos::Vec3V	p1;		//!< End of segment
		aos::FloatV	radius;
	};
}

}

// ===== 원본: physx/source/geomutils/src/common/GuBarycentricCoordinates.h
namespace px
{
namespace Gu
{
	//calculate the barycentric coorinates for a point in a segment
	EHDV void barycentricCoordinates(const aos::Vec3VArg p, 
		const aos::Vec3VArg a, 
		const aos::Vec3VArg b, 
		aos::FloatV& v);

	//calculate the barycentric coorinates for a point in a triangle
	EHDV void barycentricCoordinates(const aos::Vec3VArg p, 
		const aos::Vec3VArg a, 
		const aos::Vec3VArg b, 
		const aos::Vec3VArg c, 
		aos::FloatV& v, 
		aos::FloatV& w);

	EHDV void barycentricCoordinates(const aos::Vec3VArg v0, 
		const aos::Vec3VArg v1, 
		const aos::Vec3VArg v2,
		aos::FloatV& v, 
		aos::FloatV& w);

	EHDI aos::BoolV isValidTriangleBarycentricCoord(const aos::FloatVArg v, const aos::FloatVArg w)
	{
		using namespace aos;
		const FloatV zero = FNeg(FEps());
		const FloatV one = FAdd(FOne(), FEps());

		const BoolV con0 = BAnd(FIsGrtrOrEq(v, zero), FIsGrtrOrEq(one, v));
		const BoolV con1 = BAnd(FIsGrtrOrEq(w, zero), FIsGrtrOrEq(one, w));
		const BoolV con2 = FIsGrtr(one, FAdd(v, w));
		return BAnd(con0, BAnd(con1, con2));
	}

	EHDI aos::BoolV isValidTriangleBarycentricCoord2(const aos::Vec4VArg vwvw)
	{
		using namespace aos;
		const Vec4V eps = V4Splat(FEps());
		const Vec4V zero = V4Neg(eps);
		const Vec4V one = V4Add(V4One(), eps);

		const Vec4V v0v1v0v1 = V4PermXZXZ(vwvw);
		const Vec4V w0w1w0w1 = V4PermYWYW(vwvw);

		const BoolV con0 = BAnd(V4IsGrtrOrEq(v0v1v0v1, zero), V4IsGrtrOrEq(one, v0v1v0v1));
		const BoolV con1 = BAnd(V4IsGrtrOrEq(w0w1w0w1, zero), V4IsGrtrOrEq(one, w0w1w0w1));
		const BoolV con2 = V4IsGrtr(one, V4Add(v0v1v0v1, w0w1w0w1));
		return BAnd(con0, BAnd(con1, con2));
	}

} // namespace Gu

}

// ===== 원본: physx/source/geomutils/src/common/GuBarycentricCoordinates.cpp
using namespace px;
using namespace aos;

EHDI void Gu::barycentricCoordinates(const Vec3VArg p, const Vec3VArg a, const Vec3VArg b, FloatV& v)
{
	const Vec3V v0 = V3Sub(a, p);
	const Vec3V v1 = V3Sub(b, p);
	const Vec3V d = V3Sub(v1, v0);
	const FloatV denominator = V3Dot(d, d);
	const FloatV numerator = V3Dot(V3Neg(v0), d);
	const FloatV zero = FZero();
	const FloatV denom = FSel(FIsGrtr(denominator, zero), FRecip(denominator), zero);
	v = FMul(numerator, denom);
}

EHDI void Gu::barycentricCoordinates(const aos::Vec3VArg p, const aos::Vec3VArg a, const aos::Vec3VArg b, const aos::Vec3VArg c, aos::FloatV& v, aos::FloatV& w)
{
	const Vec3V ab = V3Sub(b, a);
	const Vec3V ac = V3Sub(c, a);

	const Vec3V n = V3Cross(ab, ac);

	const VecCrossV crossA = V3PrepareCross(V3Sub(a, p));
	const VecCrossV crossB = V3PrepareCross(V3Sub(b, p));
	const VecCrossV crossC = V3PrepareCross(V3Sub(c, p));
	const Vec3V bCrossC = V3Cross(crossB, crossC);
	const Vec3V cCrossA = V3Cross(crossC, crossA);
	const Vec3V aCrossB = V3Cross(crossA, crossB);

	const FloatV va = V3Dot(n, bCrossC);//edge region of BC, signed area rbc, u = S(rbc)/S(abc) for a
	const FloatV vb = V3Dot(n, cCrossA);//edge region of AC, signed area rac, v = S(rca)/S(abc) for b
	const FloatV vc = V3Dot(n, aCrossB);//edge region of AB, signed area rab, w = S(rab)/S(abc) for c
	const FloatV totalArea = FAdd(va, FAdd(vb, vc));
	const FloatV zero = FZero();
	const FloatV denom = FSel(FIsEq(totalArea, zero), zero, FRecip(totalArea));
	v = FMul(vb, denom);
	w = FMul(vc, denom);	
}

//	v0 = b - a;
//	v1 = c - a;
//	v2 = p - a;
EHDI void Gu::barycentricCoordinates(const Vec3VArg v0, const Vec3VArg v1, const Vec3VArg v2, FloatV& v, FloatV& w)
{
	const FloatV d00 = V3Dot(v0, v0);
	const FloatV d01 = V3Dot(v0, v1);
	const FloatV d11 = V3Dot(v1, v1);
	const FloatV d20 = V3Dot(v2, v0);
	const FloatV d21 = V3Dot(v2, v1);
	const FloatV denom = FRecip(FSub(FMul(d00,d11), FMul(d01, d01)));
	v = FMul(FSub(FMul(d11, d20), FMul(d01, d21)), denom);
	w = FMul(FSub(FMul(d00, d21), FMul(d01, d20)), denom);
}

// ===== 원본: physx/source/geomutils/src/common/GuSeparatingAxes.h
namespace px
{
namespace Gu
{
	// PT: this is a number of axes. Multiply by sizeof(PxVec3) for size in bytes.
	#define SEP_AXIS_FIXED_MEMORY	256

	// This class holds a list of potential separating axes.
	// - the orientation is irrelevant so V and -V should be the same vector
	// - the scale is irrelevant so V and n*V should be the same vector
	// - a given separating axis should appear only once in the class
	class SeparatingAxes
	{
	public:
		EHDI SeparatingAxes() : mNbAxes(0)	{}

		EHDV bool addAxis(const PxVec3& axis);

		EHD const PxVec3* getAxes() const
		{
			return mAxes;
		}

		EHD PxU32 getNumAxes() const
		{
			return mNbAxes;
		}

		EHD void reset()
		{
			mNbAxes = 0;
		}

	private:
		PxU32	mNbAxes;
		PxVec3	mAxes[SEP_AXIS_FIXED_MEMORY];
	};

	enum PxcSepAxisType
	{
		SA_NORMAL0,		// Normal of object 0
		SA_NORMAL1,		// Normal of object 1
		SA_EE			// Cross product of edges
	};

}
}

// ===== 원본: physx/source/geomutils/src/common/GuSeparatingAxes.cpp
using namespace px;

EHDI bool Gu::SeparatingAxes::addAxis(const PxVec3& axis)
{
	PxU32 numAxes = getNumAxes();
	const PxVec3*  axes = getAxes();
	const PxVec3*  axes_end = axes + numAxes;
	while(axes<axes_end)
	{
		if(PxAbs(axis.dot(*axes))>0.9999f)
			return false;
		axes++;
	}

	if(mNbAxes<SEP_AXIS_FIXED_MEMORY)
	{
		mAxes[mNbAxes++] = axis;
		return true;
	}

	return false;
}

// ===== 원본: physx/source/geomutils/src/gjk/GuGJKType.h
namespace px
{
namespace Gu
{
	class ConvexHullV;
	class ConvexHullNoScaleV;
	class BoxV;

	template <typename Convex> struct ConvexGeom		{ typedef Convex Type;				};
	template <> struct ConvexGeom<ConvexHullV>			{ typedef ConvexHullV Type;			};
	template <> struct ConvexGeom<ConvexHullNoScaleV>	{ typedef ConvexHullNoScaleV Type;	};
	template <> struct ConvexGeom<BoxV>					{ typedef BoxV Type;				};

	struct GjkConvex
	{
										EHDI GjkConvex(const ConvexV& convex) : mConvex(convex)	{}
		EHDI virtual							~GjkConvex()										{}

		EHD aos::FloatV		getMinMargin()		const { return mConvex.getMinMargin();		} 
		EHD aos::BoolV		isMarginEqRadius()	const { return mConvex.isMarginEqRadius();	}
		EHD bool			getMarginIsRadius()	const { return mConvex.getMarginIsRadius();	}
		EHD	aos::FloatV		getMargin()			const { return mConvex.getMargin();			}

		template <typename Convex>
		EHD const Convex&	getConvex()			const { return static_cast<const Convex&>(mConvex); }

		EHDI virtual aos::Vec3V	supportPoint(const PxI32 index)					const	{ return doVirtualSupportPoint(index);	}
		EHDI virtual aos::Vec3V	support(const aos::Vec3VArg v)					const	{ return doVirtualSupport(v);			}
		EHDI virtual aos::Vec3V	support(const aos::Vec3VArg dir, PxI32& index)	const	{ return doVirtualSupport(dir, index);	}
		EHDI virtual aos::FloatV	getSweepMargin()								const	{ return doVirtualGetSweepMargin();		}
		EHDV virtual aos::Vec3V	getCenter()										const = 0;

	protected:
		const ConvexV& mConvex;

	private:
		// PT: following functions call the v-table. I think this curious pattern is needed for the de-virtualization
		// approach used in the GJK code, combined with the GuGJKTest file that is only used externally.
		EHDI aos::Vec3V doVirtualSupportPoint(const PxI32 index)					const	{ return supportPoint(index);	}
		EHDI aos::Vec3V doVirtualSupport(const aos::Vec3VArg v)					const	{ return support(v);			}
		EHDI aos::Vec3V doVirtualSupport(const aos::Vec3VArg dir, PxI32& index)	const	{ return support(dir, index);	}
		EHDI aos::FloatV doVirtualGetSweepMargin()								const	{ return getSweepMargin();		}

		//E_NOCOPY(GjkConvex)
		GjkConvex& operator = (const GjkConvex&);
	};

	template <typename Convex>
	struct LocalConvex : public GjkConvex
	{
		EHDI LocalConvex(const Convex& convex) : GjkConvex(convex){}

		// PT: I think we omit the virtual on purpose here, for the de-virtualization approach, i.e. the code calls these
		// functions directly (bypassing the v-table). In fact I think we don't need virtuals at all here, it's probably
		// only for external cases and/or cases where we want to reduce code size.
		// The mix of virtual/force-inline/inline below is confusing/unclear.

		// GjkConvex
		EHD	aos::Vec3V	supportPoint(const PxI32 index)					const	{ return getConvex<Convex>().supportPoint(index);		}
						EHDI aos::Vec3V	support(const aos::Vec3VArg v)					const	{ return getConvex<Convex>().supportLocal(v);			}
						EHDI aos::Vec3V	support(const aos::Vec3VArg dir, PxI32& index)	const	{ return getConvex<Convex>().supportLocal(dir, index);	}
		EHDI		aos::FloatV	getSweepMargin()								const	{ return getConvex<Convex>().getSweepMargin();			}
		EHDI virtual			aos::Vec3V	getCenter()										const	{ return getConvex<Convex>().getCenter();				}
		//~GjkConvex

		//ML: we can't force inline function, otherwise win modern will throw compiler error
		EHDI LocalConvex<typename ConvexGeom<Convex>::Type > getGjkConvex() const
		{
			return LocalConvex<typename ConvexGeom<Convex>::Type >(static_cast<const typename ConvexGeom<Convex>::Type&>(GjkConvex::mConvex));
		}

		typedef LocalConvex<typename ConvexGeom<Convex>::Type > ConvexGeomType;

		typedef Convex	Type;

	private:
		//E_NOCOPY(LocalConvex<Convex>)
		LocalConvex<Convex>& operator = (const LocalConvex<Convex>&);
	};

	template <typename Convex>
	struct RelativeConvex : public GjkConvex
	{
		EHDI RelativeConvex(const Convex& convex, const aos::PxMatTransformV& aToB) : GjkConvex(convex), mAToB(aToB), mAToBTransposed(aToB)
		{
			aos::V3Transpose(mAToBTransposed.rot.col0, mAToBTransposed.rot.col1, mAToBTransposed.rot.col2);
		}

		// GjkConvex
		EHD	aos::Vec3V	supportPoint(const PxI32 index)					const	{ return mAToB.transform(getConvex<Convex>().supportPoint(index));					}
						EHDI aos::Vec3V	support(const aos::Vec3VArg v)					const	{ return getConvex<Convex>().supportRelative(v, mAToB, mAToBTransposed);			}
						EHDI aos::Vec3V	support(const aos::Vec3VArg dir, PxI32& index)	const	{ return getConvex<Convex>().supportRelative(dir, mAToB, mAToBTransposed, index);	}
		EHDI		aos::FloatV	getSweepMargin()								const	{ return getConvex<Convex>().getSweepMargin();										}
		EHDI virtual			aos::Vec3V	getCenter()										const	{ return mAToB.transform(getConvex<Convex>().getCenter());							}
		//~GjkConvex

		EHD const aos::PxMatTransformV&	getRelativeTransform()			const	{ return mAToB;	}

		//ML: we can't force inline function, otherwise win modern will throw compiler error
		EHDI RelativeConvex<typename ConvexGeom<Convex>::Type > getGjkConvex() const
		{
			return RelativeConvex<typename ConvexGeom<Convex>::Type >(static_cast<const typename ConvexGeom<Convex>::Type&>(GjkConvex::mConvex), mAToB);
		}

		typedef RelativeConvex<typename ConvexGeom<Convex>::Type > ConvexGeomType;

		typedef Convex	Type;

	private:
		//E_NOCOPY(RelativeConvex<Convex>)
		RelativeConvex<Convex>& operator = (const RelativeConvex<Convex>&);

		const aos::PxMatTransformV& mAToB;
		aos::PxMatTransformV mAToBTransposed;	// PT: precomputed mAToB transpose (because 'rotate' is faster than 'rotateInv')
	};

}
}

// ===== 원본: physx/source/geomutils/src/gjk/GuGJKUtil.h
/*
	This file is used to avoid the inner loop cross DLL calls
*/
namespace px
{
namespace Gu
{

enum GjkStatus
{
	GJK_NON_INTERSECT,	// two shapes doesn't intersect
	GJK_CLOSE,			// two shapes doesn't intersect and gjk algorithm will return closest point information
	GJK_CONTACT,		// two shapes overlap within margin 
	GJK_UNDEFINED,		// undefined status
	GJK_DEGENERATE,		// gjk can't converge

	EPA_CONTACT,		// two shapes intersect
	EPA_DEGENERATE,		// epa can't converge
	EPA_FAIL			// epa fail to construct an initial polygon to work with 
};

struct GjkOutput
{
public:
	EHDI GjkOutput()
	{
		using namespace aos;
		closestA = closestB = normal = V3Zero();
		penDep = FZero();
	}
	aos::Vec3V closestA;
	aos::Vec3V closestB;
	aos::Vec3V normal;
	aos::Vec3V searchDir;
	aos::FloatV penDep;
};

}//Gu
}//physx

// ===== 원본: physx/source/geomutils/src/gjk/GuGJKSimplex.h
#define EPX_GJK_INLINE EHDI
#define EPX_GJK_FORCE_INLINE EHD


namespace px
{
namespace Gu
{
	EHDV EPX_NOALIAS aos::Vec3V closestPtPointTetrahedron(aos::Vec3V*  Q, aos::Vec3V*  A, aos::Vec3V*  B, PxU32& size);

	EHDV EPX_NOALIAS aos::Vec3V closestPtPointTetrahedron(aos::Vec3V*  Q, aos::Vec3V*  A, aos::Vec3V*  B, PxI32*  aInd, PxI32*  bInd, 
		PxU32& size);

	EPX_NOALIAS EHD aos::BoolV PointOutsideOfPlane4(const aos::Vec3VArg _a, const aos::Vec3VArg _b, const aos::Vec3VArg _c, const aos::Vec3VArg _d)
	{
		using namespace aos;
		
		const Vec4V zero = V4Load(0.f);

		const Vec3V ab = V3Sub(_b, _a);
		const Vec3V ac = V3Sub(_c, _a);
		const Vec3V ad = V3Sub(_d, _a);
		const Vec3V bd = V3Sub(_d, _b);
		const Vec3V bc = V3Sub(_c, _b);

		const Vec3V v0 = V3Cross(ab, ac);
		const Vec3V v1 = V3Cross(ac, ad);
		const Vec3V v2 = V3Cross(ad, ab);
		const Vec3V v3 = V3Cross(bd, bc);

		const FloatV signa0 = V3Dot(v0, _a);
		const FloatV signa1 = V3Dot(v1, _a);
		const FloatV signa2 = V3Dot(v2, _a);
		const FloatV signd3 = V3Dot(v3, _a);

		const FloatV signd0 = V3Dot(v0, _d);
		const FloatV signd1 = V3Dot(v1, _b);
		const FloatV signd2 = V3Dot(v2, _c);
		const FloatV signa3 = V3Dot(v3, _b);

		const Vec4V signa = V4Merge(signa0, signa1, signa2, signa3);
		const Vec4V signd = V4Merge(signd0, signd1, signd2, signd3);
		return V4IsGrtrOrEq(V4Mul(signa, signd), zero);//same side, outside of the plane
	}

	EPX_NOALIAS EHD aos::Vec3V closestPtPointSegment(aos::Vec3V*  Q, PxU32& size)
	{
		using namespace aos;
		const Vec3V a = Q[0];
		const Vec3V b = Q[1];

		//const Vec3V origin = V3Zero();
		const FloatV zero = FZero();
		const FloatV one = FOne();

		//Test degenerated case
		const Vec3V ab = V3Sub(b, a);
		const FloatV denom = V3Dot(ab, ab);
		const Vec3V ap = V3Neg(a);//V3Sub(origin, a);
		const FloatV nom = V3Dot(ap, ab);
		const BoolV con = FIsGrtrOrEq(FEps(), denom);//FIsEq(denom, zero);
		//TODO - can we get rid of this branch? The problem is size, which isn't a vector!
		if(BAllEqTTTT(con))
		{
			size = 1;
			return Q[0];
		}

	/*	const PxU32 count = BAllEq(con, bTrue);
		size = 2 - count;*/
		
		const FloatV tValue = FClamp(FDiv(nom, denom), zero, one);
		return V3ScaleAdd(ab, tValue, a);
	}

	EHD void getClosestPoint(const aos::Vec3V*  Q, const aos::Vec3V*  A, const aos::Vec3V*  B, const aos::Vec3VArg closest, aos::Vec3V& closestA, aos::Vec3V& closestB, const PxU32 size)
	{
		using namespace aos;

		switch(size)
		{
		case 1:
			{
				closestA = A[0];
				closestB = B[0];
				break;
			}
		case 2:
			{
				FloatV v;
				barycentricCoordinates(closest, Q[0], Q[1], v);
				const Vec3V av = V3Sub(A[1], A[0]);
				const Vec3V bv = V3Sub(B[1], B[0]);
				closestA = V3ScaleAdd(av, v, A[0]);
				closestB = V3ScaleAdd(bv, v, B[0]);
				
				break;
			}
		case 3:
			{
				//calculate the Barycentric of closest point p in the mincowsky sum
				FloatV v, w;
				barycentricCoordinates(closest, Q[0], Q[1], Q[2], v, w);

				const Vec3V av0 = V3Sub(A[1], A[0]);
				const Vec3V av1 = V3Sub(A[2], A[0]);
				const Vec3V bv0 = V3Sub(B[1], B[0]);
				const Vec3V bv1 = V3Sub(B[2], B[0]);

				closestA = V3Add(A[0], V3Add(V3Scale(av0, v), V3Scale(av1, w)));
				closestB = V3Add(B[0], V3Add(V3Scale(bv0, v), V3Scale(bv1, w)));
			}
		};
	}

	EPX_NOALIAS EPX_GJK_FORCE_INLINE aos::FloatV closestPtPointTriangleBaryCentric(const aos::Vec3VArg a, const aos::Vec3VArg b, const aos::Vec3VArg c, 
		PxU32*  indices, PxU32& size, aos::Vec3V& closestPt)
	{
		using namespace aos;

		size = 3;
		const FloatV zero = FZero();
		const FloatV eps = FEps();
		
		const Vec3V ab = V3Sub(b, a);
		const Vec3V ac = V3Sub(c, a);

		const Vec3V n = V3Cross(ab, ac);
		//ML: if the shape is oblong, the degeneracy test sometime can't catch the degeneracy in the tetraheron. Therefore, we need to make sure we still can ternimate with the previous
		//triangle by returning the maxinum distance. 
		const FloatV nn = V3Dot(n, n);
		if (FAllEq(nn, zero))
			return FMax();

		//const FloatV va = FNegScaleSub(d5, d4, FMul(d3, d6));//edge region of BC
		//const FloatV vb = FNegScaleSub(d1, d6, FMul(d5, d2));//edge region of AC
		//const FloatV vc = FNegScaleSub(d3, d2, FMul(d1, d4));//edge region of AB

		//const FloatV va = V3Dot(n, V3Cross(b, c));//edge region of BC, signed area rbc, u = S(rbc)/S(abc) for a
		//const FloatV vb = V3Dot(n, V3Cross(c, a));//edge region of AC, signed area rac, v = S(rca)/S(abc) for b
		//const FloatV vc = V3Dot(n, V3Cross(a, b));//edge region of AB, signed area rab, w = S(rab)/S(abc) for c

		const VecCrossV crossA = V3PrepareCross(a);
		const VecCrossV crossB = V3PrepareCross(b);
		const VecCrossV crossC = V3PrepareCross(c);
		const Vec3V bCrossC = V3Cross(crossB, crossC);
		const Vec3V cCrossA = V3Cross(crossC, crossA);
		const Vec3V aCrossB = V3Cross(crossA, crossB);

		const FloatV va = V3Dot(n, bCrossC);//edge region of BC, signed area rbc, u = S(rbc)/S(abc) for a
		const FloatV vb = V3Dot(n, cCrossA);//edge region of AC, signed area rac, v = S(rca)/S(abc) for b
		const FloatV vc = V3Dot(n, aCrossB);//edge region of AB, signed area rab, w = S(rab)/S(abc) for c

		const BoolV isFacePoints = BAnd(FIsGrtrOrEq(va, zero), BAnd(FIsGrtrOrEq(vb, zero), FIsGrtrOrEq(vc, zero)));

		//face region
		if(BAllEqTTTT(isFacePoints))
		{	
			const FloatV t = FDiv(V3Dot(n, a), nn);
			const Vec3V q = V3Scale(n, t);
			closestPt = q;
			return V3Dot(q, q);
		}

		const Vec3V ap = V3Neg(a);
		const Vec3V bp = V3Neg(b);
		const Vec3V cp = V3Neg(c);

		const FloatV d1 = V3Dot(ab, ap); //  snom
		const FloatV d2 = V3Dot(ac, ap); //  tnom
		const FloatV d3 = V3Dot(ab, bp); // -sdenom
		const FloatV d4 = V3Dot(ac, bp); //  unom = d4 - d3
		const FloatV d5 = V3Dot(ab, cp); //  udenom = d5 - d6
		const FloatV d6 = V3Dot(ac, cp); // -tdenom

		const FloatV unom = FSub(d4, d3);
		const FloatV udenom = FSub(d5, d6);

		size = 2;
		//check if p in edge region of AB
		const BoolV con30 = FIsGrtrOrEq(zero, vc);
		const BoolV con31 = FIsGrtrOrEq(d1, zero);
		const BoolV con32 = FIsGrtrOrEq(zero, d3);
		const BoolV con3 = BAnd(con30, BAnd(con31, con32));//edge AB region
		if(BAllEqTTTT(con3))
		{
			const FloatV toRecipAB = FSub(d1, d3);
			const FloatV recipAB = FSel(FIsGrtr(FAbs(toRecipAB), eps), FRecip(toRecipAB), zero);
			const FloatV t = FMul(d1, recipAB);
			const Vec3V q = V3ScaleAdd(ab, t, a);
			closestPt = q;
			return V3Dot(q, q);
		}
	
		//check if p in edge region of BC
		const BoolV con40 = FIsGrtrOrEq(zero, va);
		const BoolV con41 = FIsGrtrOrEq(d4, d3);
		const BoolV con42 = FIsGrtrOrEq(d5, d6);
		const BoolV con4 = BAnd(con40, BAnd(con41, con42)); //edge BC region
		if(BAllEqTTTT(con4))
		{
			const Vec3V bc = V3Sub(c, b);
			const FloatV toRecipBC = FAdd(unom, udenom);
			const FloatV recipBC = FSel(FIsGrtr(FAbs(toRecipBC), eps), FRecip(toRecipBC), zero);
			const FloatV t = FMul(unom, recipBC);
			indices[0] = indices[1];
			indices[1] = indices[2];
			const Vec3V q = V3ScaleAdd(bc, t, b);
			closestPt = q;
			return V3Dot(q, q);
		}
		
		//check if p in edge region of AC
		const BoolV con50 = FIsGrtrOrEq(zero, vb);
		const BoolV con51 = FIsGrtrOrEq(d2, zero);
		const BoolV con52 = FIsGrtrOrEq(zero, d6);
	
		const BoolV con5 = BAnd(con50, BAnd(con51, con52));//edge AC region
		if(BAllEqTTTT(con5))
		{
			const FloatV toRecipAC = FSub(d2, d6);
			const FloatV recipAC = FSel(FIsGrtr(FAbs(toRecipAC), eps), FRecip(toRecipAC), zero);
			const FloatV t = FMul(d2, recipAC);
			indices[1]=indices[2];
			const Vec3V q = V3ScaleAdd(ac, t, a);
			closestPt = q;
			return V3Dot(q, q);
		}

		size = 1;
		//check if p in vertex region outside a
		const BoolV con00 = FIsGrtrOrEq(zero, d1); // snom <= 0
		const BoolV con01 = FIsGrtrOrEq(zero, d2); // tnom <= 0
		const BoolV con0 = BAnd(con00, con01); // vertex region a
		if(BAllEqTTTT(con0))
		{
			closestPt = a;
			return V3Dot(a, a);
		}

		//check if p in vertex region outside b
		const BoolV con10 = FIsGrtrOrEq(d3, zero);
		const BoolV con11 = FIsGrtrOrEq(d3, d4);
		const BoolV con1 = BAnd(con10, con11); // vertex region b
		if(BAllEqTTTT(con1))
		{
			indices[0] = indices[1];
			closestPt = b;
			return V3Dot(b, b);
		}
		
		//p is in vertex region outside c
		indices[0] = indices[2];
		closestPt = c;
		return V3Dot(c, c);
	}

	EPX_NOALIAS EPX_GJK_FORCE_INLINE aos::Vec3V closestPtPointTriangle(aos::Vec3V*  Q, aos::Vec3V* A, aos::Vec3V* B, PxU32& size)
	{
		using namespace aos;

		size = 3;
	
		const FloatV eps2 = FLoad(EPX_EPS_REAL * EPX_EPS_REAL);
		const Vec3V a = Q[0];
		const Vec3V b = Q[1];
		const Vec3V c = Q[2];
		const Vec3V ab = V3Sub(b, a);
		const Vec3V ac = V3Sub(c, a);
		const Vec3V signArea = V3Cross(ab, ac);//0.5*(abXac)
		const FloatV area = V3Dot(signArea, signArea);
		if(FAllGrtrOrEq(eps2, area))
		{
			//degenerate
			size = 2;
			return closestPtPointSegment(Q, size);
		}

		PxU32 _size;
		PxU32 indices[3]={0, 1, 2};
		Vec3V closestPt;
		closestPtPointTriangleBaryCentric(a, b, c, indices, _size, closestPt);

		if(_size != 3)
		{
			const Vec3V q0 = Q[indices[0]]; const Vec3V q1 = Q[indices[1]];
			const Vec3V a0 = A[indices[0]]; const Vec3V a1 = A[indices[1]];
			const Vec3V b0 = B[indices[0]]; const Vec3V b1 = B[indices[1]];

			Q[0] = q0; Q[1] = q1;
			A[0] = a0; A[1] = a1;
			B[0] = b0; B[1] = b1;

			size = _size;
		}

		return closestPt;
	}

	EPX_NOALIAS EPX_GJK_FORCE_INLINE aos::Vec3V closestPtPointTriangle(aos::Vec3V*  Q, aos::Vec3V* A, aos::Vec3V* B, PxI32*  aInd, PxI32*  bInd, 
		PxU32& size)
	{
		using namespace aos;

		size = 3;
	
		const FloatV eps2 = FLoad(EPX_EPS_REAL * EPX_EPS_REAL);

		const Vec3V a = Q[0];
		const Vec3V b = Q[1];
		const Vec3V c = Q[2];
		const Vec3V ab = V3Sub(b, a);
		const Vec3V ac = V3Sub(c, a);
		const Vec3V signArea = V3Cross(ab, ac);//0.5*(abXac)
		const FloatV area = V3Dot(signArea, signArea);
		if(FAllGrtrOrEq(eps2, area))
		{
			//degenerate
			size = 2;
			return closestPtPointSegment(Q, size);
		}

		PxU32 _size;
		PxU32 indices[3]={0, 1, 2};
		Vec3V closestPt;
		closestPtPointTriangleBaryCentric(a, b, c, indices, _size, closestPt);

		if(_size != 3)
		{
			const Vec3V q0 = Q[indices[0]]; const Vec3V q1 = Q[indices[1]];
			const Vec3V a0 = A[indices[0]]; const Vec3V a1 = A[indices[1]];
			const Vec3V b0 = B[indices[0]]; const Vec3V b1 = B[indices[1]];
			const PxI32 aInd0 = aInd[indices[0]]; const PxI32 aInd1 = aInd[indices[1]];
			const PxI32 bInd0 = bInd[indices[0]]; const PxI32 bInd1 = bInd[indices[1]];

			Q[0] = q0; Q[1] = q1;
			A[0] = a0; A[1] = a1;
			B[0] = b0; B[1] = b1;
			aInd[0] = aInd0; aInd[1] = aInd1;
			bInd[0] = bInd0; bInd[1] = bInd1;

			size = _size;
		}

		return closestPt;
	}

	EPX_NOALIAS EHD aos::Vec3V GJKCPairDoSimplex(aos::Vec3V*  Q, aos::Vec3V*  A, aos::Vec3V*  B, const aos::Vec3VArg support, 
		PxU32& size)
	{
		using namespace aos;

		//const PxU32 tempSize = size;
		//calculate a closest from origin to the simplex
		switch(size)
		{
		case 1:
			{
				return support;
			}
		case 2:
			{
			return closestPtPointSegment(Q, size);
			}
		case 3:
			{
			return closestPtPointTriangle(Q, A, B, size);
			}
		case 4:
			return closestPtPointTetrahedron(Q, A, B, size);
		default:
			E_ASSERT(0);
		}
		return support;
	}

	EPX_NOALIAS EHD aos::Vec3V GJKCPairDoSimplex(aos::Vec3V*  Q, aos::Vec3V*  A, aos::Vec3V*  B, PxI32*  aInd, PxI32*  bInd, 
		const aos::Vec3VArg support, PxU32& size)
	{
		using namespace aos;

		//const PxU32 tempSize = size;
		//calculate a closest from origin to the simplex
		switch(size)
		{
		case 1:
			{
				return support;
			}
		case 2:
			{
			return closestPtPointSegment(Q, size);
			}
		case 3:
			{
			return closestPtPointTriangle(Q, A, B, aInd, bInd, size);
			}
		case 4:
			return closestPtPointTetrahedron(Q, A, B, aInd, bInd, size);
		default:
			E_ASSERT(0);
		}
		return support;
	}
}

}

// ===== 원본: physx/source/geomutils/src/gjk/GuGJKSimplex.cpp
namespace px
{
namespace Gu
{

	using namespace aos;

	EHDI static Vec3V getClosestPtPointTriangle(Vec3V*  Q, const BoolVArg bIsOutside4, PxU32* indices, PxU32& size)
	{
		FloatV bestSqDist = FMax();
		
		PxU32 _indices[3] = {0, 1, 2};

		Vec3V closestPt = V3Zero();
	
		if(BAllEqTTTT(BGetX(bIsOutside4)))
		{
			//use the original indices, size, v and w
			bestSqDist = closestPtPointTriangleBaryCentric(Q[0], Q[1], Q[2], indices, size, closestPt);
		}

		if(BAllEqTTTT(BGetY(bIsOutside4)))
		{

			PxU32 _size = 3;
			_indices[0] = 0; _indices[1] = 2; _indices[2] = 3; 
			Vec3V tClosestPt;
			const FloatV sqDist = closestPtPointTriangleBaryCentric(Q[0], Q[2], Q[3], _indices, _size, tClosestPt);

			const BoolV con = FIsGrtr(bestSqDist, sqDist);
			if(BAllEqTTTT(con))
			{
				closestPt = tClosestPt;
				bestSqDist = sqDist;

				indices[0] = _indices[0];
				indices[1] = _indices[1];
				indices[2] = _indices[2];

				size = _size;
			}
		}

		if(BAllEqTTTT(BGetZ(bIsOutside4)))
		{
			PxU32 _size = 3;
			
			_indices[0] = 0; _indices[1] = 3; _indices[2] = 1; 

			Vec3V tClosestPt;
			const FloatV sqDist = closestPtPointTriangleBaryCentric(Q[0], Q[3], Q[1], _indices, _size, tClosestPt);
			
			const BoolV con = FIsGrtr(bestSqDist, sqDist);
			if(BAllEqTTTT(con))
			{
				closestPt = tClosestPt;
				bestSqDist = sqDist;

				indices[0] = _indices[0];
				indices[1] = _indices[1];
				indices[2] = _indices[2];

				size = _size;
			}

		}

		if(BAllEqTTTT(BGetW(bIsOutside4)))
		{
	

			PxU32 _size = 3;
			_indices[0] = 1; _indices[1] = 3; _indices[2] = 2; 
			Vec3V tClosestPt;
			const FloatV sqDist = closestPtPointTriangleBaryCentric(Q[1], Q[3], Q[2], _indices, _size, tClosestPt);

			const BoolV con = FIsGrtr(bestSqDist, sqDist);

			if(BAllEqTTTT(con))
			{
				closestPt = tClosestPt;
				bestSqDist = sqDist;

				indices[0] = _indices[0];
				indices[1] = _indices[1];
				indices[2] = _indices[2];

				size = _size;
			}
		}

		return closestPt;
	}

	EHDI EPX_NOALIAS Vec3V closestPtPointTetrahedron(Vec3V*  Q, Vec3V*  A, Vec3V*  B, PxU32& size)
	{
		
		const FloatV eps = FLoad(1e-4f);
		const Vec3V a = Q[0];
		const Vec3V b = Q[1];
		const Vec3V c = Q[2];  
		const Vec3V d = Q[3];

		//degenerated
		const Vec3V ab = V3Sub(b, a);
		const Vec3V ac = V3Sub(c, a);
		const Vec3V n = V3Normalize(V3Cross(ab, ac));
		const FloatV signDist = V3Dot(n, V3Sub(d, a));
		if(FAllGrtr(eps, FAbs(signDist)))
		{
			size = 3;
			return closestPtPointTriangle(Q, A, B, size);
		}

		const BoolV bIsOutside4 = PointOutsideOfPlane4(a, b, c, d);

		if(BAllEqFFFF(bIsOutside4))
		{
			//All inside
			return V3Zero();
		}

		PxU32 indices[3] = {0, 1, 2};
		
		const Vec3V closest = getClosestPtPointTriangle(Q, bIsOutside4, indices, size);

		const Vec3V q0 = Q[indices[0]]; const Vec3V q1 = Q[indices[1]]; const Vec3V q2 = Q[indices[2]];
		const Vec3V a0 = A[indices[0]]; const Vec3V a1 = A[indices[1]]; const Vec3V a2 = A[indices[2]];
		const Vec3V b0 = B[indices[0]]; const Vec3V b1 = B[indices[1]]; const Vec3V b2 = B[indices[2]];
		Q[0] = q0; Q[1] = q1; Q[2] = q2;
		A[0] = a0; A[1] = a1; A[2] = a2;
		B[0] = b0; B[1] = b1; B[2] = b2; 

		return closest;
	}

	EHDI EPX_NOALIAS Vec3V closestPtPointTetrahedron(Vec3V*  Q, Vec3V*  A, Vec3V*  B, PxI32*  aInd, PxI32*  bInd, PxU32& size)
	{
		
		const FloatV eps = FLoad(1e-4f);
		const Vec3V zeroV = V3Zero();
		
		const Vec3V a = Q[0];
		const Vec3V b = Q[1];
		const Vec3V c = Q[2];
		const Vec3V d = Q[3];

		//degenerated
		const Vec3V ab = V3Sub(b, a);
		const Vec3V ac = V3Sub(c, a);
		const Vec3V n = V3Normalize(V3Cross(ab, ac));
		const FloatV signDist = V3Dot(n, V3Sub(d, a));
		if(FAllGrtr(eps, FAbs(signDist)))
		{
			size = 3;
			return closestPtPointTriangle(Q, A, B, aInd, bInd, size);
		}

		const BoolV bIsOutside4 = PointOutsideOfPlane4(a, b, c, d);

		if(BAllEqFFFF(bIsOutside4))
		{
			//All inside
			return zeroV;
		}

		PxU32 indices[3] = {0, 1, 2};
		const Vec3V closest = getClosestPtPointTriangle(Q, bIsOutside4, indices, size);

		const Vec3V q0 = Q[indices[0]]; const Vec3V q1 = Q[indices[1]]; const Vec3V q2 = Q[indices[2]];
		const Vec3V a0 = A[indices[0]]; const Vec3V a1 = A[indices[1]]; const Vec3V a2 = A[indices[2]];
		const Vec3V b0 = B[indices[0]]; const Vec3V b1 = B[indices[1]]; const Vec3V b2 = B[indices[2]];
		const PxI32 _aInd0 = aInd[indices[0]]; const PxI32 _aInd1 = aInd[indices[1]]; const PxI32 _aInd2 = aInd[indices[2]];
		const PxI32 _bInd0 = bInd[indices[0]]; const PxI32 _bInd1 = bInd[indices[1]]; const PxI32 _bInd2 = bInd[indices[2]];
		Q[0] = q0; Q[1] = q1; Q[2] = q2;
		A[0] = a0; A[1] = a1; A[2] = a2;
		B[0] = b0; B[1] = b1; B[2] = b2; 
		aInd[0] = _aInd0; aInd[1] = _aInd1; aInd[2] = _aInd2;
		bInd[0] = _bInd0; bInd[1] = _bInd1; bInd[2] = _bInd2;

		return closest;
	}
}

}

// ===== 원본: physx/source/geomutils/src/gjk/GuGJKPenetration.h
#define	GJK_VALIDATE 0

namespace px
{
namespace Gu
{
	class ConvexV;

	EHD void assignWarmStartValue(PxU8*  aIndices, PxU8*  bIndices, PxU8& size_, PxI32*  aInd, PxI32*  bInd, PxU32 size )
	{
		if(aIndices)
		{
			E_ASSERT(bIndices);
			size_ = PxTo8(size);
			for(PxU32 i=0; i<size; ++i)
			{
				aIndices[i] = PxTo8(aInd[i]);
				bIndices[i] = PxTo8(bInd[i]);
			}
		}
	}

	EHD void validateDuplicateVertex(const aos::Vec3V* Q, const aos::Vec3VArg support, const PxU32 size)
	{
		using namespace aos;

		const FloatV eps = FEps();
		//Get rid of the duplicate point
		BoolV match = BFFFF();
		for(PxU32 na = 0; na < size; ++na)
		{
			Vec3V dif = V3Sub(Q[na], support);
			match = BOr(match, FIsGrtr(eps, V3Dot(dif, dif)));	
		}

		//we have duplicate code
		if(BAllEqTTTT(match))
		{
			E_ASSERT(0);
		}
	}

	//*Each convex has
	//*         a support function
	//*         a margin	- if the shape is sphere/capsule, margin is the radius
	//*         a minMargin - some percentage of margin, which is used to determine the termination condition for gjk
	
	//*We'll report:
	//*         GJK_NON_INTERSECT if the minimum distance between the shapes is greater than the sum of the margins and the the contactDistance
	//*         EPA_CONTACT if shapes overlap. We treat sphere/capsule as a point/a segment and we shrunk other shapes by 10% of margin
	//*         GJK_CONTACT if the algorithm converges, and the distance between the shapes is less than the sum of the margins plus the contactDistance. In this case we return the closest points found
	//*         GJK_DEGENERATE if the algorithm doesn't converge, we return this flag to indicate the normal and closest point we return might not be accurated
	template<typename ConvexA, typename ConvexB >
	 EHDI GjkStatus gjkPenetration(const ConvexA& a, const ConvexB& b, const aos::Vec3VArg initialSearchDir, const aos::FloatVArg contactDist, const bool takeCoreShape, 
		PxU8*  aIndices, PxU8*  bIndices, aos::Vec3V*  aPoints, aos::Vec3V*  bPoints, PxU8& warmStartSize,
		GjkOutput& output)
	{
		using namespace aos;

		//ML: eps is the threshold that uses to determine whether two (shrunk) shapes overlap. We calculate eps2 based on 10% of the minimum margin of two shapes
		const FloatV minMargin = FMin(a.ConvexA::getMinMargin(), b.ConvexB::getMinMargin());
		const FloatV eps = FMul(minMargin, FLoad(0.1f));

		//const FloatV eps2 = FMul(_eps2, _eps2);
		//ML: epsRel2 is the square of 0.01. This is used to scale the square distance of a closest point to origin to determine whether two shrunk shapes overlap in the margin, but
		//they don't overlap.
		//const FloatV epsRel2 = FMax(FLoad(0.0001), eps2); 

		// ML:epsRel is square value of 1.5% which applied to the distance of a closest point(v) to the origin.
		// If |v|- v/|v|.dot(w) < epsRel*|v|=>(|v|*(1-epsRel) < v/|v|.dot(w)),
		// two shapes are clearly separated, GJK terminate and return non intersect.
		// This adjusts the termination condition based on the length of v
		// which avoids ill-conditioned terminations. 
		const FloatV epsRel = FLoad(0.000225f);//1.5%.
		const FloatV relDif = FSub(FOne(), epsRel);

		const FloatV zero = FZero();
		
		//capsule/sphere will have margin which is its radius
		const FloatV marginA = a.getMargin();
		const FloatV marginB = b.getMargin();

		const BoolV aQuadratic = a.isMarginEqRadius();
		const BoolV bQuadratic = b.isMarginEqRadius();
		const FloatV tMarginA = FSel(aQuadratic, marginA, zero);
		const FloatV tMarginB = FSel(bQuadratic, marginB, zero);

		const FloatV sumMargin = FAdd(tMarginA, tMarginB);
		const FloatV sumExpandedMargin = FAdd(sumMargin, contactDist);
		
		FloatV dist = FMax();
		FloatV prevDist = dist;
		const Vec3V zeroV = V3Zero();
		Vec3V prevClos = zeroV;

		const BoolV bTrue = BTTTT();
		BoolV bNotTerminated = bTrue;
		BoolV bNotDegenerated = bTrue;
		Vec3V closest;

		Vec3V Q[4];
		Vec3V* A = aPoints;
		Vec3V* B = bPoints;
		PxI32 aInd[4];
		PxI32 bInd[4];
		Vec3V supportA = zeroV, supportB = zeroV, support=zeroV;
		Vec3V v;

		PxU32 size = 0;//_size;
		

		//ML: if _size!=0, which means we pass in the previous frame simplex so that we can warm-start the simplex. 
		//In this case, GJK will normally terminate in one iteration
		if(warmStartSize != 0)
		{
			for(PxU32 i=0; i<warmStartSize; ++i)
			{
				aInd[i] = aIndices[i];
				bInd[i] = bIndices[i];

				//de-virtualize
				supportA = a.ConvexA::supportPoint(aIndices[i]);
				supportB = b.ConvexB::supportPoint(bIndices[i]);

				support = V3Sub(supportA, supportB);

				A[size] =  supportA;
				B[size] =  supportB;
				Q[size++] =	support;
			}

			//run simplex solver to determine whether the point is closest enough so that gjk can terminate
			closest = GJKCPairDoSimplex(Q, A, B, aInd, bInd, support, size);
			dist = V3Length(closest);
			//sDist = V3Dot(closest, closest);
			v = V3ScaleInv(closest, dist);
			prevDist = dist;
			prevClos = closest;
			
			bNotTerminated = FIsGrtr(dist, eps);
		}
		else
		{
			//const Vec3V _initialSearchDir = V3Sub(a.getCenter(), b.getCenter());
			closest = V3Sel(FIsGrtr(V3Dot(initialSearchDir, initialSearchDir), zero), initialSearchDir, V3UnitX());
			v = V3Normalize(closest);
		}
		
		
		// ML : termination condition
		//(1)two shapes overlap. GJK will terminate based on sq(v) < eps2 and indicate that two shapes are overlapping.
		//(2)two shapes are separated. If sq(vw) > sqMargin * sq(v), which means the original objects do not intesect, GJK terminate with GJK_NON_INTERSECT. 
		//(3)two shapes don't overlap. However, they interect within margin distance. if sq(v)- vw < epsRel2*sq(v), this means the shrunk shapes interect in the margin, 
		//   GJK terminate with GJK_CONTACT.
		while(BAllEqTTTT(bNotTerminated))
		{
			//prevDist, prevClos are used to store the previous iteration's closest point and the square distance from the closest point
			//to origin in Mincowski space
			prevDist = dist;
			prevClos = closest;

			//de-virtualize
			supportA = a.ConvexA::support(V3Neg(closest), aInd[size]);
			supportB = b.ConvexB::support(closest, bInd[size]);

			//calculate the support point
			support = V3Sub(supportA, supportB);

			const FloatV vw = V3Dot(v, support);
			if(FAllGrtr(vw, sumExpandedMargin))
			{
				assignWarmStartValue(aIndices, bIndices, warmStartSize, aInd, bInd, size);
				return GJK_NON_INTERSECT;
			}
			
			//if(FAllGrtr(FMul(epsRel, dist), FSub(dist, vw)))
			if(FAllGrtr(vw, FMul(dist, relDif)))
			{
				assignWarmStartValue(aIndices, bIndices, warmStartSize, aInd, bInd, size);
				E_ASSERT(FAllGrtr(dist, FEps()));
				//const Vec3V n = V3ScaleInv(closest, dist);//normalise
				output.normal = v; 
				Vec3V closA, closB;
				getClosestPoint(Q, A, B, closest, closA, closB, size);
				//ML: if one of the shape is sphere/capsule and the takeCoreShape flag is true, the contact point for sphere/capsule will be the sphere center or a point in the
				//capsule segment. This will increase the stability for the manifold recycling code. Otherwise, we will return a contact point on the surface for sphere/capsule
				//while the takeCoreShape flag is set to be false
				if(takeCoreShape)
				{
					output.closestA = closA;
					output.closestB = closB;
					output.penDep = dist;
					
				}
				else
				{
					//This is for capsule/sphere want to take the surface point. For box/convex,
					//we need to get rid of margin 
					output.closestA = V3NegScaleSub(v, tMarginA, closA);
					output.closestB = V3ScaleAdd(v, tMarginB, closB);
					output.penDep = FSub(dist, sumMargin);
				}

				return GJK_CONTACT;
			}

			A[size] = supportA;
			B[size] = supportB;
			Q[size++]=support;
			E_ASSERT(size <= 4);

			//calculate the closest point between two convex hull
			closest = GJKCPairDoSimplex(Q, A, B, aInd, bInd, support, size);

			dist = V3Length(closest);
			v = V3ScaleInv(closest, dist);

			bNotDegenerated = FIsGrtr(prevDist, dist);
			bNotTerminated = BAnd(FIsGrtr(dist, eps), bNotDegenerated);
		}

		if(BAllEqFFFF(bNotDegenerated))
		{
			assignWarmStartValue(aIndices, bIndices, warmStartSize, aInd, bInd, size-1);
			
			//Reset back to older closest point
			dist = prevDist;
			closest = prevClos;//V3Sub(closA, closB);
			Vec3V closA, closB;
			getClosestPoint(Q, A, B, closest, closA, closB, size);

			//E_ASSERT(FAllGrtr(dist, FEps()));
			const Vec3V n = V3ScaleInv(prevClos, prevDist);//normalise
			output.normal = n;

			output.searchDir = v;

			if(takeCoreShape)
			{
				output.closestA = closA;
				output.closestB = closB;
				output.penDep = dist;
			}
			else
			{
				//This is for capsule/sphere want to take the surface point. For box/convex,
				//we need to get rid of margin
				output.closestA = V3NegScaleSub(n, tMarginA, closA);
				output.closestB = V3ScaleAdd(n, tMarginB, closB);
				output.penDep = FSub(dist, sumMargin);
				if (FAllGrtrOrEq(sumMargin, dist))
					return GJK_CONTACT;
			}
			
			return GJK_DEGENERATE;  
		}
		else 
		{
			//this two shapes are deeply intersected with each other, we need to use EPA algorithm to calculate MTD
			assignWarmStartValue(aIndices, bIndices, warmStartSize, aInd, bInd, size);
			return EPA_CONTACT;
			
		}
	}
	template<typename ConvexA, typename ConvexB >
	 EHDI GjkStatus gjkPenetration(const ConvexA& a, const ConvexB& b, const aos::Vec3VArg initialSearchDir, const aos::FloatVArg contactDist, const bool takeCoreShape,
		PxU8*  aIndices, PxU8*  bIndices, PxU8& warmStartSize,
		GjkOutput& output)
	{
		aos::Vec3V aPoints[4], bPoints[4];
		return gjkPenetration(a, b, initialSearchDir, contactDist, takeCoreShape, aIndices, bIndices, aPoints, bPoints, warmStartSize, output);
	}
	template<typename ConvexA, typename ConvexB >
	 EHDI GjkStatus gjkPenetration(const ConvexA& a, const ConvexB& b, const aos::Vec3VArg initialSearchDir, const aos::FloatVArg contactDist, const bool takeCoreShape,
		aos::Vec3V*  aPoints, aos::Vec3V*  bPoints, PxU8& warmStartSize,
		GjkOutput& output)
	{
		PxU8 aIndices[4], bIndices[4];
		return gjkPenetration(a, b, initialSearchDir, contactDist, takeCoreShape, aIndices, bIndices, aPoints, bPoints, warmStartSize, output);
	}
}//Gu

}//physx

// ===== 원본: physx/source/common/src/CmPriorityQueue.h
namespace px
{
namespace Cm
{
	template<class Element, class Comparator = PxLess<Element> >
	class PriorityQueueBase : protected Comparator // inherit so that stateless comparators take no space
	{
	public:
		EHDI PriorityQueueBase(const Comparator& less, Element* elements) : Comparator(less), mHeapSize(0), mDataPtr(elements)
		{
		}
		
		EHDI ~PriorityQueueBase() 
		{
		}
		
		//! Get the element with the highest priority
		EHD const Element top() const
		{
			return mDataPtr[0];
		}

		//! Get the element with the highest priority
		EHD Element top()
		{
			return mDataPtr[0];
		}
		
		//! Check to whether the priority queue is empty
		EHD bool empty() const
		{
			return (mHeapSize == 0);
		}
		
		//! Empty the priority queue
		EHD void clear()
		{
			mHeapSize = 0;
		}  

		//! Insert a new element into the priority queue. Only valid when size() is less than Capacity
		EHD void push(const Element& value)
		{
			PxU32 newIndex;
			PxU32 parentIndex = parent(mHeapSize);

			for (newIndex = mHeapSize; newIndex > 0 && compare(value, mDataPtr[parentIndex]); newIndex = parentIndex, parentIndex = parent(newIndex)) 
			{
				mDataPtr[ newIndex ] = mDataPtr[parentIndex];
			}
			mDataPtr[newIndex] = value; 
			mHeapSize++;
			E_ASSERT(valid());
		}

		//! Delete the highest priority element. Only valid when non-empty.
		EHD Element pop()
		{
			E_ASSERT(mHeapSize > 0);
			PxU32 i, child;
			//try to avoid LHS
			PxU32 tempHs = mHeapSize-1;
			mHeapSize = tempHs;
			Element min = mDataPtr[0];
			Element last = mDataPtr[tempHs];
			
			for (i = 0; (child = left(i)) < tempHs; i = child) 
			{
				/* Find highest priority child */
				const PxU32 rightChild = child + 1;
			
				child += ((rightChild < tempHs) & compare((mDataPtr[rightChild]), (mDataPtr[child]))) ? 1 : 0;

				if(compare(last, mDataPtr[child]))
					break;

				mDataPtr[i] = mDataPtr[child];
			}
			mDataPtr[ i ] = last;
			
			E_ASSERT(valid());
			return min;
		} 

		//! Make sure the priority queue sort all elements correctly
		EHDI bool valid() const
		{
			const Element& min = mDataPtr[0];
			for(PxU32 i=1; i<mHeapSize; ++i)
			{
				if(compare(mDataPtr[i], min))
					return false;
			}

			return true;
		}

		//! Return number of elements in the priority queue
		EHDI PxU32 size() const
		{
			return mHeapSize;
		}

	protected:

		PxU32 mHeapSize;
		Element* mDataPtr;
		
		EHD bool compare(const Element& a, const Element& b) const
		{
			return Comparator::operator()(a,b);
		}

		static EHD PxU32 left(PxU32 nodeIndex) 
		{
			return (nodeIndex << 1) + 1;
		}
		
		static EHD PxU32 parent(PxU32 nodeIndex) 
		{
			return (nodeIndex - 1) >> 1;
		}
	private:
		PriorityQueueBase<Element, Comparator>& operator = (const PriorityQueueBase<Element, Comparator>);
	};

	template <typename Element, PxU32 Capacity, typename Comparator>
	class InlinePriorityQueue : public PriorityQueueBase<Element, Comparator>
	{
		Element mData[Capacity];
	public:
		EHDI InlinePriorityQueue(const Comparator& less = Comparator()) : PriorityQueueBase<Element, Comparator>(less, mData)
		{
		}

		EHD void push(Element& elem)
		{
			E_ASSERT(this->mHeapSize < Capacity);
			PriorityQueueBase<Element, Comparator>::push(elem);
		}
	private:
		InlinePriorityQueue<Element, Capacity, Comparator>& operator = (const InlinePriorityQueue<Element, Capacity, Comparator>);
	};

	template <typename Element, typename Comparator, typename Alloc = typename px::PxAllocatorTraits<Element>::Type>
	class PriorityQueue : public PriorityQueueBase<Element, Comparator>, protected Alloc
	{
		PxU32 mCapacity;
	public:
		EHDI PriorityQueue(const Comparator& less = Comparator(), PxU32 initialCapacity = 0, Alloc alloc = Alloc()) 
			: PriorityQueueBase<Element, Comparator>(less, NULL), Alloc(alloc), mCapacity(initialCapacity)
		{
			if(initialCapacity > 0)
				this->mDataPtr = reinterpret_cast<Element*>(Alloc::allocate(sizeof(Element)*initialCapacity, EPX_FL));
		}

		EHDI ~PriorityQueue()
		{
			if(this->mDataPtr)
				this->deallocate(this->mDataPtr);
		}

		EHD void push(Element& elem)
		{
			if(this->mHeapSize == mCapacity)
			{
				reserve((this->mHeapSize+1)*2);
			}
			PriorityQueueBase<Element, Comparator>::push(elem);
		}

		EHD PxU32 capacity()
		{
			return mCapacity;
		}

		EHD void reserve(const PxU32 newCapacity)
		{
			if(newCapacity > mCapacity)
			{
				Element* newElems = reinterpret_cast<Element*>(Alloc::allocate(sizeof(Element)*newCapacity, EPX_FL));
				if(this->mDataPtr)
				{
					px::PxMemCopy(newElems, this->mDataPtr, sizeof(Element) * this->mHeapSize);
					Alloc::deallocate(this->mDataPtr);
				}
				this->mDataPtr = newElems;
				mCapacity = newCapacity;
			}
		}

	private:
		PriorityQueue<Element, Comparator, Alloc>& operator = (const PriorityQueue<Element, Comparator, Alloc>);
	};
	
}
}

// ===== 원본: physx/source/common/src/CmIDPool.h
namespace px
{
namespace Cm
{
	template<class FreeBuffer>
	class IDPoolBase : public PxUserAllocated
	{
	protected:
		PxU32			mCurrentID;
		FreeBuffer		mFreeIDs;
	public:
        EHDI IDPoolBase() : mCurrentID(0)	{}

		EHDI void	freeID(PxU32 id)
		{
			// Allocate on first call
			// Add released ID to the array of free IDs
			if(id == (mCurrentID - 1))
				--mCurrentID;
			else
				mFreeIDs.pushBack(id);
		}

		EHDI void	freeAll()
		{
			mCurrentID = 0;
			mFreeIDs.clear();
		}

		EHDI PxU32	getNewID()
		{
			// If recycled IDs are available, use them
			const PxU32 size = mFreeIDs.size();
			if(size)
			{
				// Recycle last ID
				return mFreeIDs.popBack();
			}
			// Else create a new ID
			return mCurrentID++;
		}

		EHDI PxU32 getNumUsedID()	const
		{
			return mCurrentID - mFreeIDs.size();
		}

		EHDI PxU32 getMaxID() const
		{
			return mCurrentID;
		}

	};

	//This class extends IDPoolBase. This is mainly used for when it is unsafe for the application to free the id immediately so that it can
	//defer the free process until it is safe to do so
	template<class FreeBuffer>
	class DeferredIDPoolBase : public IDPoolBase<FreeBuffer>
	{
		FreeBuffer mDeferredFreeIDs;
	public:
		//release an index into the deferred list
		EHDI void	deferredFreeID(PxU32 id)
		{
			mDeferredFreeIDs.pushBack(id);
		}

		//release the deferred indices into the free list
		EHDI void processDeferredIds()
		{
			const PxU32 deferredFreeIDCount = mDeferredFreeIDs.size();
			for(PxU32 a = 0; a < deferredFreeIDCount;++a)
			{
				IDPoolBase<FreeBuffer>::freeID(mDeferredFreeIDs[a]);
			}
			mDeferredFreeIDs.clear();
		}

		//release all indices
		EHDI void	freeAll()
		{
			mDeferredFreeIDs.clear();
			IDPoolBase<FreeBuffer>::freeAll();
		}

		EHDI PxU32 getNumUsedID()	const
		{
			return IDPoolBase<FreeBuffer>::getNumUsedID() - mDeferredFreeIDs.size();
		}

		EHDI FreeBuffer& getDeferredFreeIDs() { return mDeferredFreeIDs; }
	};

	//This is spu friendly fixed size array
	template <typename T, uint32_t N>
	class InlineFixedArray
	{
		T mArr[N];
		PxU32 mSize;
	public:

		EHDI InlineFixedArray() : mSize(0)
		{
		}

		EHDI ~InlineFixedArray(){}

		EHDI void pushBack(const T& t)
		{
			E_ASSERT(mSize < N);
			mArr[mSize++] = t;
		}

		EHDI T popBack()
		{
			E_ASSERT(mSize > 0);
			return mArr[--mSize];
		}

		EHDI void clear() { mSize = 0; }

		EHDI T& operator [] (PxU32 index) { E_ASSERT(index < N); return mArr[index]; }

		EHDI const T& operator [] (PxU32 index) const { E_ASSERT(index < N); return mArr[index]; }

		EHDI PxU32 size() const { return mSize; }
	};

	//Fix size IDPool
	template<PxU32 Capacity>
	class InlineIDPool : public IDPoolBase<InlineFixedArray<PxU32, Capacity> >
	{
	public:
		EHDI PxU32 getNumRemainingIDs()
		{
			return Capacity - this->getNumUsedID();
		}
	};

	//Dynamic resize IDPool
	class IDPool : public IDPoolBase<PxArray<PxU32> >
	{
	};


	//This class is used to recycle indices. It supports deferred release, so that until processDeferredIds is called,
	//released indices will not be reallocated. This class will fail if the calling code request more id than the InlineDeferredIDPoll
	//has. It is the calling code's responsibility to ensure that this does not happen.
	template<PxU32 Capacity>
	class InlineDeferredIDPool : public DeferredIDPoolBase<InlineFixedArray<PxU32, Capacity> >
	{
	public:
		EHDI PxU32 getNumRemainingIDs()
		{
			return Capacity - IDPoolBase< InlineFixedArray<PxU32, Capacity> >::getNumUsedID();
		}
	};

	//Dynamic resize DeferredIDPool
	class DeferredIDPool : public DeferredIDPoolBase<PxArray<PxU32> >
	{

	};

} // namespace Cm

}

// ===== 원본: physx/source/geomutils/src/gjk/GuEPAFacet.h
#define EPX_EPA_FORCE_INLINE EHD

#define EPA_DEBUG	0

namespace px
{
#define MaxEdges 32
#define MaxFacets 64
#define MaxSupportPoints 64

namespace Gu
{
	E_GTABLE(PxU32, lookUp, 3, {1, 2, 0})
	
	EHD PxU32 incMod3(PxU32 i) { return lookUp_p()[i]; } 
	
	class EdgeBuffer;
	class Edge;
	typedef Cm::InlineDeferredIDPool<MaxFacets>	EPAFacetManager;

	class Facet	
	{
	public:
	
		EHDI Facet()
		{
		}

		EHD Facet(const PxU32 _i0, const PxU32 _i1, const PxU32 _i2)
			: m_obsolete(false), m_inHeap(false)
		{
			m_indices[0]= PxToI8(_i0);
			m_indices[1]= PxToI8(_i1);
			m_indices[2]= PxToI8(_i2);
			 
			m_adjFacets[0] = m_adjFacets[1] = m_adjFacets[2] = NULL;
			m_adjEdges[0] = m_adjEdges[1] = m_adjEdges[2] = -1;
		}


		EHD void invalidate()
		{
			m_adjFacets[0] = m_adjFacets[1] = m_adjFacets[2] = NULL;
			m_adjEdges[0] = m_adjEdges[1] = m_adjEdges[2] = -1;
		}

		EHD bool Valid()
		{
			return (m_adjFacets[0] != NULL) & (m_adjFacets[1] != NULL) & (m_adjFacets[2] != NULL);
		}

		EHD PxU32 operator[](const PxU32 i) const 
		{ 
			return PxU32(m_indices[i]);
		} 

		//create ajacency information
		EHDV bool link(const PxU32 edge0, Facet*   facet, const PxU32 edge1);

		EHD bool isObsolete() const { return m_obsolete; }

		//calculate the signed distance from a point to a plane
		EHD aos::FloatV getPlaneDist(const aos::Vec3VArg p, const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf) const; 

		//check to see whether the triangle is a valid triangle, calculate plane normal and plane distance at the same time
		EHDV aos::BoolV isValid2(const PxU32 i0, const PxU32 i1, const PxU32 i2, const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf, 
		const aos::FloatVArg upper);

		//return the absolute value for the plane distance from origin 
		EHD aos::FloatV getPlaneDist() const 
		{ 
			return aos::FLoad(m_planeDist);
		}

		//return the plane normal
		EHD aos::Vec3V getPlaneNormal()const
		{
			return m_planeNormal; 
		}

		//calculate the closest points for a shape pair
		EHDV void getClosestPoint(const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf, aos::Vec3V& closestA, 
			aos::Vec3V& closestB);

		//calculate the closest points for a shape pair
		//void getClosestPoint(const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf, aos::FloatV& v, aos::FloatV& w);
		
		//performs a flood fill over the boundary of the current polytope. 
		EHDV void silhouette(const aos::Vec3VArg w, const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf, EdgeBuffer& edgeBuffer, EPAFacetManager& manager);
		

		//m_planeDist is positive
		EHDI bool operator <(const Facet& b) const
		{
			return m_planeDist < b.m_planeDist;
		}
	 
		//store all the boundary facets for the new polytope in the edgeBuffer and free indices when an old facet isn't part of the boundary anymore
		EHD void silhouette(const PxU32 index, const aos::Vec3VArg w, const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf, EdgeBuffer& edgeBuffer, 
			EPAFacetManager& manager);

		aos::Vec3V m_planeNormal;																										//16
		PxF32 m_planeDist;
		
		Facet*  m_adjFacets[3]; //the triangle adjacent to edge i in this triangle												//32
		PxI8 m_adjEdges[3]; //the edge connected with the corresponding triangle															//35
		PxI8 m_indices[3]; //the index of vertices of the triangle																			//38
		bool m_obsolete; //a flag to denote whether the triangle are still part of the bundeary of the new polytope							//39
		bool m_inHeap;	//a flag to indicate whether the triangle is in the heap															//40
		PxU8 m_FacetId;																														//41																									//73

	};

	
	class Edge 
	{
	public:
		EHD Edge() {}
		EHD Edge(Facet *  facet, const PxU32 index) : m_facet(facet), m_index(index) {}
		EHD Edge(const Edge& other) : m_facet(other.m_facet), m_index(other.m_index){}

		EHD Edge& operator = (const Edge& other)
		{
			m_facet = other.m_facet;
			m_index = other.m_index;
			return *this;
		}

		EHD Facet *getFacet() const { return m_facet; }
		EHD PxU32 getIndex() const { return m_index; }

		//get out the associated start vertex index in this edge from the facet
		EHD PxU32 getSource() const
		{
			E_ASSERT(m_index < 3);
			return (*m_facet)[m_index];
		}

		//get out the associated end vertex index in this edge from the facet
		EHD PxU32 getTarget() const
		{
			E_ASSERT(m_index < 3);
			return (*m_facet)[incMod3(m_index)];
		}

		Facet*  m_facet;
		PxU32 m_index;
	};


	class EdgeBuffer
	{
	public:
		EHDI EdgeBuffer() : m_Size(0), m_OverFlow(false)
		{
		}

		EHDI Edge* Insert(Facet*   facet, const PxU32 index)
		{
			if (m_Size < MaxEdges)
			{
				Edge* pEdge = &m_pEdges[m_Size++];
				pEdge->m_facet = facet;
				pEdge->m_index = index;
				return pEdge;
			}
			m_OverFlow = true;
			return NULL;
		}

		EHDI Edge* Get(const PxU32 index)
		{
			E_ASSERT(index < m_Size);
			return &m_pEdges[index];
		}

		EHDI PxU32 Size()
		{
			return m_Size;
		}

		EHDI bool IsValid()
		{
			return m_Size > 0 && !m_OverFlow;
		}

		EHDI void MakeEmpty()
		{
			m_Size = 0;
			m_OverFlow = false;
		}

		Edge	m_pEdges[MaxEdges];
		PxU32	m_Size;
		bool	m_OverFlow;
	};

	//ML: calculate MTD points for a shape pair
	EHD void Facet::getClosestPoint(const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf, aos::Vec3V& closestA, 
		aos::Vec3V& closestB)
	{
		using namespace aos;

		const Vec3V pa0(aBuf[m_indices[0]]);
		const Vec3V pa1(aBuf[m_indices[1]]);
		const Vec3V pa2(aBuf[m_indices[2]]);

		const Vec3V pb0(bBuf[m_indices[0]]);
		const Vec3V pb1(bBuf[m_indices[1]]);
		const Vec3V pb2(bBuf[m_indices[2]]);

		const Vec3V p0 = V3Sub(pa0, pb0);
		const Vec3V p1 = V3Sub(pa1, pb1);
		const Vec3V p2 = V3Sub(pa2, pb2);

		const Vec3V v0 = V3Sub(p1, p0);
		const Vec3V v1 = V3Sub(p2, p0);

		const Vec3V closestP = V3Scale(m_planeNormal, FLoad(m_planeDist));
		const Vec3V v2 = V3Sub(closestP, p0);
		//calculate barycentric coordinates
		const FloatV d00 = V3Dot(v0, v0);
		const FloatV d01 = V3Dot(v0, v1);
		const FloatV d11 = V3Dot(v1, v1);

		const FloatV d20 = V3Dot(v2, v0);
		const FloatV d21 = V3Dot(v2, v1);

		const FloatV det = FNegScaleSub(d01, d01, FMul(d00, d11));//FSub( FMul(v1dv1, v2dv2), FMul(v1dv2, v1dv2) ); // non-negative
		const FloatV recip = FSel(FIsGrtr(det, FEps()), FRecip(det), FZero());

		const FloatV lambda1 = FMul(FNegScaleSub(d01, d21, FMul(d11, d20)), recip);
		const FloatV lambda2 = FMul(FNegScaleSub(d01, d20, FMul(d00, d21)), recip);


		const FloatV u = FSub(FOne(), FAdd(lambda1, lambda2));
		closestA = V3ScaleAdd(pa0, u, V3ScaleAdd(pa1, lambda1, V3Scale(pa2, lambda2)));
		closestB = V3ScaleAdd(pb0, u, V3ScaleAdd(pb1, lambda1, V3Scale(pb2, lambda2)));
	}

	//ML: create adjacency informations for both facets
	EHD bool Facet::link(const PxU32 edge0, Facet *  facet, const PxU32 edge1) 
	{
		m_adjFacets[edge0] = facet;
		m_adjEdges[edge0] = PxToI8(edge1);
		facet->m_adjFacets[edge1] = this;
		facet->m_adjEdges[edge1] = PxToI8(edge0);

		return (m_indices[edge0] == facet->m_indices[incMod3(edge1)]) && (m_indices[incMod3(edge0)] == facet->m_indices[edge1]);
	} 

}

}

// ===== 원본: physx/source/geomutils/src/gjk/GuEPA.h
namespace px
{
  
namespace Gu
{
	//ML: The main entry point for EPA.
	// 
	//This function returns one of three status codes:
	//(1)EPA_FAIL:	the algorithm failed to create a valid polytope(the origin wasn't inside the polytope) from the input simplex.
	//(2)EPA_CONTACT : the algorithm found the MTD and converged successfully.
	//(3)EPA_DEGENERATE: the algorithm cannot make further progress and the result is unknown.

	EHDV GjkStatus epaPenetration(	const GjkConvex& a,						//convex a in the space of convex b
								const GjkConvex& b, 					//convex b							   
								const PxU8*  aInd,			//warm start index for convex a to create an initial simplex
								const PxU8*  bInd,			//warm start index for convex b to create an initial simplex
								const PxU8 size,						//number of warm-start indices						    
								const bool takeCoreShape,				//indicates whether we take support point from the core shape or surface of capsule/sphere
								const aos::FloatV tolerenceLength,		//the length of meter
								GjkOutput& output);						//result					

	EHDV GjkStatus epaPenetration(	const GjkConvex& a,						//convex a in the space of convex b
								const GjkConvex& b, 					//convex b							   
								const aos::Vec3V*  aPnt,		//warm start point for convex a to create an initial simplex
								const aos::Vec3V*  bPnt,		//warm start point for convex b to create an initial simplex
								const PxU8 size,						//number of warm-start indices						    
								const bool takeCoreShape,				//indicates whether we take support point from the core shape or surface of capsule/sphere
								const aos::FloatV tolerenceLength,		//the length of meter
								GjkOutput& output);						//result					
}

}

// ===== 원본: physx/source/geomutils/src/gjk/GuEPA.cpp
namespace px
{
namespace Gu
{
	using namespace aos;

	class ConvexV;

	struct FacetDistanceComparator
	{
		EHDI bool operator()(const Facet* left, const Facet* right) const
		{
			return *left < *right;
		}
	};



	class EPA
	{	
	public:
		EHDI EPA(){}	
		EHDV GjkStatus PenetrationDepth(const GjkConvex& a, const GjkConvex& b, const aos::Vec3V*  A, const aos::Vec3V*  B, const PxU8 size, const bool takeCoreShape, 
			const FloatV tolerenceLength, GjkOutput& output);
		EHDV bool expandPoint(const GjkConvex& a, const GjkConvex& b, PxI32& numVerts, const FloatVArg upperBound);
		EHDV bool expandSegment(const GjkConvex& a, const GjkConvex& b, PxI32& numVerts, const FloatVArg upperBound);
		EHDV bool expandTriangle(PxI32& numVerts, const FloatVArg upperBound);

		EHDV Facet* addFacet(const PxU32 i0, const PxU32 i1, const PxU32 i2, const aos::FloatVArg upper);
	
		EHDV bool originInTetrahedron(const aos::Vec3VArg p1, const aos::Vec3VArg p2, const aos::Vec3VArg p3, const aos::Vec3VArg p4);

		Cm::InlinePriorityQueue<Facet*, MaxFacets, FacetDistanceComparator> heap;
		aos::Vec3V aBuf[MaxSupportPoints];
		aos::Vec3V bBuf[MaxSupportPoints];
		Facet facetBuf[MaxFacets];
		EdgeBuffer edgeBuffer;
		EPAFacetManager facetManager;

	private:
		E_NOCOPY(EPA)
	};

	EHD bool EPA::originInTetrahedron(const aos::Vec3VArg p1, const aos::Vec3VArg p2, const aos::Vec3VArg p3, const aos::Vec3VArg p4)
	{
		using namespace aos;
		return BAllEqFFFF(PointOutsideOfPlane4(p1, p2, p3, p4)) == 1;
	}

	static EHD void doSupport(const GjkConvex& a, const GjkConvex& b, const aos::Vec3VArg dir, aos::Vec3V& supportA, aos::Vec3V& supportB, aos::Vec3V& support)
	{
		const Vec3V tSupportA = a.support(V3Neg(dir));
		const Vec3V tSupportB = b.support(dir);
		//avoid LHS
		supportA = tSupportA;
		supportB = tSupportB;
		support = V3Sub(tSupportA, tSupportB);
	}

	EHDI GjkStatus epaPenetration(const GjkConvex& a, const GjkConvex& b, const PxU8*  aInd, const PxU8*  bInd, const PxU8 size, 
		const bool takeCoreShape, const FloatV tolerenceLength, GjkOutput& output)
	{

		E_ASSERT(size > 0 && size <=4);
	
		Vec3V A[4];
		Vec3V B[4];
	
		//ML: we construct a simplex based on the gjk simplex indices
		for(PxU32 i=0; i<size; ++i)
		{
			A[i] = a.supportPoint(aInd[i]);
			B[i] = b.supportPoint(bInd[i]);
		}
		
		EPA epa;

		return epa.PenetrationDepth(a, b, A, B, size, takeCoreShape, tolerenceLength, output);
	}

	EHDI GjkStatus epaPenetration(const GjkConvex& a, const GjkConvex& b, const aos::Vec3V*  aPnt, const aos::Vec3V*  bPnt, const PxU8 size,
		const bool takeCoreShape, const FloatV tolerenceLength, GjkOutput& output)
	{

		E_ASSERT(size > 0 && size <=4);
	
		const Vec3V* A = aPnt;
		const Vec3V* B = bPnt;
		
		EPA epa;

		return epa.PenetrationDepth(a, b, A, B, size, takeCoreShape, tolerenceLength, output);
	}

	//ML: this function returns the signed distance of a point to a plane
	EHD aos::FloatV Facet::getPlaneDist(const aos::Vec3VArg p, const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf) const
	{
		const Vec3V pa0(aBuf[m_indices[0]]);
		const Vec3V pb0(bBuf[m_indices[0]]);
		const Vec3V p0 = V3Sub(pa0, pb0);

		return V3Dot(m_planeNormal, V3Sub(p, p0));
	}

	//ML: This function:
	// (1)calculates the distance from orign((0, 0, 0)) to a triangle plane 
	// (2) rejects triangle if the triangle is degenerate (two points are identical)
	// (3) rejects triangle to be added into the heap if the plane distance is large than upper
	EHDI aos::BoolV Facet::isValid2(const PxU32 i0, const PxU32 i1, const PxU32 i2, const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf, 
		const aos::FloatVArg upper)
	{
		using namespace aos;
		const FloatV eps = FEps();

		const Vec3V pa0(aBuf[i0]);
		const Vec3V pa1(aBuf[i1]);
		const Vec3V pa2(aBuf[i2]);

		const Vec3V pb0(bBuf[i0]);
		const Vec3V pb1(bBuf[i1]);
		const Vec3V pb2(bBuf[i2]);

		const Vec3V p0 = V3Sub(pa0, pb0);
		const Vec3V p1 = V3Sub(pa1, pb1);
		const Vec3V p2 = V3Sub(pa2, pb2);

		const Vec3V v0 = V3Sub(p1, p0);
		const Vec3V v1 = V3Sub(p2, p0);

		const Vec3V denormalizedNormal = V3Cross(v0, v1);
		FloatV norValue = V3Dot(denormalizedNormal, denormalizedNormal);
		//if norValue < eps, this triangle is degenerate
		const BoolV con = FIsGrtr(norValue, eps);
		norValue = FSel(con, norValue, FOne());
	
		const Vec3V planeNormal = V3Scale(denormalizedNormal, FRsqrt(norValue));
		const FloatV planeDist =  V3Dot(planeNormal, p0);

		m_planeNormal = planeNormal;
		FStore(planeDist, &m_planeDist);
	
		return BAnd(con, FIsGrtrOrEq(upper, planeDist));
	}

	//ML: if the triangle is valid(not degenerate and within lower and upper bound), we need to add it into the heap. Otherwise, we just return
	//the triangle so that the facet can be linked to other facets in the expanded polytope.
	EHDI Facet* EPA::addFacet(const PxU32 i0, const PxU32 i1, const PxU32 i2, const aos::FloatVArg upper)
	{
		using namespace aos;
		E_ASSERT(i0 != i1 && i0 != i2 && i1 != i2);
		//ML: we move the control in the calling code so we don't need to check weather we will run out of facets or not
		E_ASSERT(facetManager.getNumUsedID() < MaxFacets);

		const PxU32 facetId = facetManager.getNewID();
		PxPrefetchLine(&facetBuf[facetId], 128);

		Facet * facet = E_PLACEMENT_NEW(&facetBuf[facetId],Facet(i0, i1, i2));
		facet->m_FacetId = PxU8(facetId);

		const  BoolV validTriangle = facet->isValid2(i0, i1, i2, aBuf, bBuf, upper);

		if(BAllEqTTTT(validTriangle))
		{
			heap.push(facet);
			facet->m_inHeap = true;
		}
		else
		{
			facet->m_inHeap = false;
		}
		return facet;
	}

	//ML: this function performs a flood fill over the boundary of the current polytope. 
	EHDI void Facet::silhouette(const PxU32 _index, const aos::Vec3VArg w, const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf, EdgeBuffer& edgeBuffer, EPAFacetManager& manager) 
	{
		using namespace aos;
		const FloatV zero = FZero();
		Edge stack[MaxFacets];
		stack[0] = Edge(this, _index);
		PxI32 size = 1;
		while(size--)
		{
			Facet* const  f = stack[size].m_facet;
			const PxU32 index = stack[size].m_index;
			E_ASSERT(f->Valid());
			
			if(!f->m_obsolete)
			{
				//ML: if the point is above the facet, the facet has an reflex edge, which will make the polytope concave. Therefore, we need to 
				//remove this facet and make sure the expanded polytope is convex.
				const FloatV pointPlaneDist = f->getPlaneDist(w, aBuf, bBuf);
				
				if(FAllGrtr(zero, pointPlaneDist))
				{
					//ML: facet isn't visible from w (we don't have a reflex edge), this facet will be on the boundary and part of the new polytope so that
					//we will push it into our edgeBuffer
					if(!edgeBuffer.Insert(f, index))
						return;
				} 
				else 
				{
					//ML:facet is visible from w, therefore, we need to remove this facet from the heap and push its adjacent facets onto the stack
					f->m_obsolete = true; // Facet is visible from w
					const PxU32 next(incMod3(index));
					const PxU32 next2(incMod3(next));
					stack[size++] = Edge(f->m_adjFacets[next2],PxU32(f->m_adjEdges[next2]));
					stack[size++] = Edge(f->m_adjFacets[next], PxU32(f->m_adjEdges[next]));
					
					E_ASSERT(size <= MaxFacets);
					if(!f->m_inHeap)
					{
						//if the facet isn't in the heap, we can release that memory
						manager.deferredFreeID(f->m_FacetId);
					}
				}
			}
		}
	}

	//ML: this function perform flood fill for the adjancent facet and store the boundary facet into the edgeBuffer
	EHDI void Facet::silhouette(const aos::Vec3VArg w, const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf, EdgeBuffer& edgeBuffer, EPAFacetManager& manager)
	{
		m_obsolete = true;
		for(PxU32 a = 0; a < 3; ++a)
		{
			m_adjFacets[a]->silhouette(PxU32(m_adjEdges[a]), w, aBuf, bBuf, edgeBuffer, manager);
		}
	}

	EHDI bool EPA::expandPoint(const GjkConvex& a, const GjkConvex& b, PxI32& numVerts, const FloatVArg upperBound)
	{
		const Vec3V x = V3UnitX();
		Vec3V q0 = V3Sub(aBuf[0], bBuf[0]);
		Vec3V q1;
		doSupport(a, b, x, aBuf[1], bBuf[1], q1);
		if (V3AllEq(q0, q1))
			return false;
		return expandSegment(a, b, numVerts, upperBound);
	}

	//ML: this function use the segement to create a triangle
	EHDI bool EPA::expandSegment(const GjkConvex& a, const GjkConvex& b, PxI32& numVerts, const FloatVArg upperBound)
	{
		const Vec3V q0 = V3Sub(aBuf[0], bBuf[0]);
		const Vec3V q1 = V3Sub(aBuf[1], bBuf[1]);
		const Vec3V v = V3Sub(q1, q0);
		const Vec3V absV = V3Abs(v);

		const FloatV x = V3GetX(absV);
		const FloatV y = V3GetY(absV);
		const FloatV z = V3GetZ(absV);

		Vec3V axis = V3UnitX();
		const BoolV con0 = BAnd(FIsGrtr(x, y), FIsGrtr(z, y));
		if (BAllEqTTTT(con0))
		{
			axis = V3UnitY();
		}
		else if(FAllGrtr(x, z))
		{
			axis = V3UnitZ();
		}
		
		const Vec3V n = V3Normalize(V3Cross(axis, v));
		Vec3V q2;
		doSupport(a, b, n, aBuf[2], bBuf[2], q2);

		return expandTriangle(numVerts, upperBound);
	}

	EHDI bool EPA::expandTriangle(PxI32& numVerts, const FloatVArg upperBound)
	{
		numVerts = 3;

		Facet *  f0 = addFacet(0, 1, 2, upperBound);
		Facet *  f1 = addFacet(1, 0, 2, upperBound);

		if(heap.empty())
			return false;

		f0->link(0, f1, 0);
		f0->link(1, f1, 2);
		f0->link(2, f1, 1);

		return true;
	}

	//ML: this function calculates contact information. If takeCoreShape flag is true, this means the two closest points will be on the core shape used in the support functions. 
	//For example, we treat sphere/capsule as a point/segment in the support function for GJK/EPA, so that the core shape for sphere/capsule is a point/segment. For PCM, we need 
	//to take the point from the core shape because this will allows us recycle the contacts more stably. For SQ sweeps, we need to take the point on the surface of the sphere/capsule 
	//when we calculate MTD because this is what will be reported to the user. Therefore, the takeCoreShape flag will be set to be false in SQ.
	EHDI static void calculateContactInformation(const aos::Vec3V*  aBuf, const aos::Vec3V*  bBuf, Facet* facet, const GjkConvex& a, const GjkConvex& b, const bool takeCoreShape, GjkOutput& output)
	{
		const FloatV zero = FZero();
		Vec3V _pa, _pb;
		facet->getClosestPoint(aBuf, bBuf, _pa, _pb);

		//dist > 0 means two shapes are penetrated. If dist < 0(when origin isn't inside the polytope), two shapes status are unknown
		const FloatV dist = FAbs(facet->getPlaneDist());
		
		const Vec3V planeNormal = V3Neg(facet->getPlaneNormal());
		
		if(takeCoreShape)
		{
			output.closestA = _pa;
			output.closestB = _pb;
			output.normal = planeNormal;
			output.penDep = FNeg(dist);
		}
		else
		{
			//for sphere/capusule to take the surface point
			const BoolV aQuadratic = a.isMarginEqRadius();
			const BoolV bQuadratic = b.isMarginEqRadius();

			const FloatV marginA = FSel(aQuadratic, a.getMargin(), zero);
			const FloatV marginB = FSel(bQuadratic, b.getMargin(), zero);
			const FloatV sumMargin = FAdd(marginA, marginB);
			output.closestA = V3NegScaleSub(planeNormal, marginA, _pa);
			output.closestB = V3ScaleAdd(planeNormal, marginB, _pb);
			output.normal = planeNormal;
			output.penDep = FNeg(FAdd(dist, sumMargin));
		}
	}

	//ML: This function returns one of three status codes:
	//(1)EPA_FAIL:	the algorithm failed to create a valid polytope(the origin wasn't inside the polytope) from the input simplex
	//(2)EPA_CONTACT : the algorithm found the MTD and converged successfully.
	//(3)EPA_DEGENERATE: the algorithm cannot make further progress and the result is unknown.
	EHDI GjkStatus EPA::PenetrationDepth(const GjkConvex& a, const GjkConvex& b, const aos::Vec3V*  A, const aos::Vec3V*  B, const PxU8 size, const bool takeCoreShape, 
		const FloatV tolerenceLength, GjkOutput& output)
	{
	
		using namespace aos; 

		E_UNUSED(tolerenceLength);

		PxPrefetchLine(&facetBuf[0]);
		PxPrefetchLine(&facetBuf[0], 128);

		const FloatV zero = FZero();
	
		const FloatV _max = FMax();
		FloatV upper_bound(_max);
	
		aBuf[0]=A[0]; aBuf[1]=A[1]; aBuf[2]=A[2]; aBuf[3]=A[3];
		bBuf[0]=B[0]; bBuf[1]=B[1]; bBuf[2]=B[2]; bBuf[3]=B[3];

		PxI32 numVertsLocal = 0;

		heap.clear();

		//if the simplex isn't a tetrahedron, we need to construct one before we can expand it
		switch (size) 
		{
		case 1:
			{
				// Touching contact. Yes, we have a collision and the penetration will be zero
				if(!expandPoint(a, b, numVertsLocal, upper_bound))
					return EPA_FAIL;
				break;
			}
		case 2: 
			{
				// We have a line segment inside the Minkowski sum containing the
				// origin. we need to construct two back to back triangles which link to each other
				if(!expandSegment(a, b, numVertsLocal, upper_bound))
					return EPA_FAIL;
				break;
			}
		case 3: 
			{
				// We have a triangle inside the Minkowski sum containing
				// the origin. We need to construct two back to back triangles which link to each other
				if(!expandTriangle(numVertsLocal, upper_bound))
					return EPA_FAIL;

				break;
				
			}
		case 4:
			{
				//check for input face normal. All face normals in this tetrahedron should be all pointing either inwards or outwards. If all face normals are pointing outward, we are good to go. Otherwise, we need to 
				//shuffle the input vertexes and make sure all face normals are pointing outward
				const Vec3V pa0(aBuf[0]);
				const Vec3V pa1(aBuf[1]);
				const Vec3V pa2(aBuf[2]);
				const Vec3V pa3(aBuf[3]);

				const Vec3V pb0(bBuf[0]);
				const Vec3V pb1(bBuf[1]);
				const Vec3V pb2(bBuf[2]);
				const Vec3V pb3(bBuf[3]);

				const Vec3V p0 = V3Sub(pa0, pb0);
				const Vec3V p1 = V3Sub(pa1, pb1);
				const Vec3V p2 = V3Sub(pa2, pb2);
				const Vec3V p3 = V3Sub(pa3, pb3);

				const Vec3V v1 = V3Sub(p1, p0);
				const Vec3V v2 = V3Sub(p2, p0);

				const Vec3V planeNormal = V3Normalize(V3Cross(v1, v2));

				const FloatV signDist = V3Dot(planeNormal, V3Sub(p3, p0));

				if (FAllGrtr(signDist, zero))
				{
					//shuffle the input vertexes
					const Vec3V tempA0 = aBuf[2];
					const Vec3V tempB0 = bBuf[2];
					aBuf[2] = aBuf[1];
					bBuf[2] = bBuf[1];
					aBuf[1] = tempA0;
					bBuf[1] = tempB0;
				}

				Facet *  f0 = addFacet(0, 1, 2, upper_bound);
				Facet *  f1 = addFacet(0, 3, 1, upper_bound);
				Facet *  f2 = addFacet(0, 2, 3, upper_bound);
				Facet *  f3 = addFacet(1, 3, 2, upper_bound);
			
				if (heap.empty())
					return EPA_FAIL;


				f0->link(0, f1, 2);
				f0->link(1, f3, 2);
				f0->link(2, f2, 0);
				f1->link(0, f2, 2);
				f1->link(1, f3, 0);
				f2->link(1, f3, 1);
				numVertsLocal = 4;

				break;
			}
		}

	
		const FloatV minMargin = FMin(a.getMinMargin(), b.getMinMargin());
		const FloatV eps = FMul(minMargin, FLoad(0.1f));

		Facet*  facet = NULL;

		Vec3V tempa, tempb, q;

		do 
		{
	
			facetManager.processDeferredIds();
			facet = heap.pop(); //get the shortest distance triangle of origin from the list
			facet->m_inHeap = false;
			
			if (!facet->isObsolete()) 
			{
				PxPrefetchLine(edgeBuffer.m_pEdges);
				PxPrefetchLine(edgeBuffer.m_pEdges,128);
				PxPrefetchLine(edgeBuffer.m_pEdges,256);

				const Vec3V planeNormal = facet->getPlaneNormal();
				const FloatV planeDist = facet->getPlaneDist();

				tempa = a.support(planeNormal);
				tempb = b.support(V3Neg(planeNormal));

				q = V3Sub(tempa, tempb);
				
				PxPrefetchLine(&aBuf[numVertsLocal],128);
				PxPrefetchLine(&bBuf[numVertsLocal],128);

				//calculate the distance from support point to the origin along the plane normal. Because the support point is search along
				//the plane normal, which means the distance should be positive. However, if the origin isn't contained in the polytope, dist
				//might be negative
				const FloatV dist = V3Dot(q, planeNormal);

				const BoolV con0 = FIsGrtrOrEq(eps, FAbs(FSub(dist, planeDist)));

				if (BAllEqTTTT(con0))
				{
					calculateContactInformation(aBuf, bBuf, facet, a, b, takeCoreShape, output);

					if (takeCoreShape)
					{
						const FloatV toleranceEps = FMul(FLoad(1e-3f), tolerenceLength);
						const Vec3V dif = V3Sub(output.closestA, output.closestB);
						const FloatV pen = FAdd(FAbs(output.penDep), toleranceEps);
						const FloatV sqDif = V3Dot(dif, dif);
						const FloatV length = FSel(FIsGrtr(sqDif, zero), FSqrt(sqDif), zero);
						if (FAllGrtr(length, pen))
							return EPA_DEGENERATE;
					}
					return EPA_CONTACT;
				
				}

				//update the upper bound to the minimum between existing upper bound and the distance
				upper_bound = FMin(upper_bound, dist);

				aBuf[numVertsLocal]=tempa;
				bBuf[numVertsLocal]=tempb;

				const PxU32 index = PxU32(numVertsLocal++);

				// Compute the silhouette cast by the new vertex
				// Note that the new vertex is on the positive side
				// of the current facet, so the current facet will
				// not be in the polytope. Start local search
				// from this facet.

				edgeBuffer.MakeEmpty();

				facet->silhouette(q, aBuf, bBuf, edgeBuffer, facetManager);

				if (!edgeBuffer.IsValid())
				{
					calculateContactInformation(aBuf, bBuf, facet, a, b, takeCoreShape, output);
					return EPA_DEGENERATE;
				}

				Edge*  edge=edgeBuffer.Get(0);

				PxU32 bufferSize=edgeBuffer.Size();

				//check to see whether we have enough space in the facet manager to create new facets
				if(bufferSize > facetManager.getNumRemainingIDs())
				{
					calculateContactInformation(aBuf, bBuf, facet, a, b, takeCoreShape, output);
					return EPA_DEGENERATE;
				}

				Facet *firstFacet = addFacet(edge->getTarget(), edge->getSource(),index, upper_bound);
				E_ASSERT(firstFacet);
				firstFacet->link(0, edge->getFacet(), edge->getIndex());

				Facet *  lastFacet = firstFacet;

				for (PxU32 i = 1; i<bufferSize; ++i)
				{
					edge = edgeBuffer.Get(i);
					Facet*  newFacet = addFacet(edge->getTarget(), edge->getSource(), index, upper_bound);
					newFacet->link(0, edge->getFacet(), edge->getIndex());
					newFacet->link(2, lastFacet, 1);
					lastFacet = newFacet;
				}

				firstFacet->link(2, lastFacet, 1);
			}
			facetManager.freeID(facet->m_FacetId);

		}
		while((heap.size() > 0) && FAllGrtr(upper_bound, heap.top()->getPlaneDist()) && numVertsLocal != MaxSupportPoints);

		calculateContactInformation(aBuf, bBuf, facet, a, b, takeCoreShape, output);
		return EPA_DEGENERATE;
	}
}

}

// ===== 원본: physx/source/geomutils/src/pcm/GuPersistentContactManifold.h
#define PCM_LOW_LEVEL_DEBUG 0

namespace px
{

#define VISUALIZE_PERSISTENT_CONTACT 1
//This is for primitives vs primitives 
#define GU_MANIFOLD_CACHE_SIZE 4
//These are for primitives vs mesh
#define GU_SINGLE_MANIFOLD_SINGLE_POLYGONE_CACHE_SIZE 5
#define GU_SINGLE_MANIFOLD_CACHE_SIZE 6
#define GU_SPHERE_MANIFOLD_CACHE_SIZE 1
#define GU_CAPSULE_MANIFOLD_CACHE_SIZE 3
#define GU_MAX_MANIFOLD_SIZE 6	// PT: max nb of manifolds (e.g. for multi-manifolds), NOT the max size of a single manifold
#define GU_MESH_CONTACT_REDUCTION_THRESHOLD	16

#define GU_MANIFOLD_INVALID_INDEX	0xffffffff

//ML: this is used to compared with the shape's margin to decide the final tolerance used in the manifold to validate the existing contacts.
//In the case of big shape and relatively speaking small triangles in the mesh, we need to take a smaller margin. This helps because the PCM 
//recycling thresholds are proportionate to margin so it makes it less likely to discard previous contacts due to separation
#define GU_PCM_MESH_MANIFOLD_EPSILON 0.05f

class PxRenderOutput;
class PxContactBuffer;
  
namespace Gu
{
	struct GjkOutput;

E_GTABLE_DECL(PxF32, invalidateThresholds); 
E_GTABLE_DECL(PxF32, invalidateQuatThresholds);
E_GTABLE_DECL(PxF32, invalidateThresholds2); 
E_GTABLE_DECL(PxF32, invalidateQuatThresholds2);

EHDV aos::Mat33V findRotationMatrixFromZAxis(const aos::Vec3VArg to);

//This contact is used in the primitives vs primitives contact gen
class PersistentContact
{
public:
	EHDI PersistentContact()
	{
	}

	EHDI PersistentContact(aos::Vec3V _localPointA, aos::Vec3V _localPointB, aos::Vec4V _localNormalPen) : 
	mLocalPointA(_localPointA), mLocalPointB(_localPointB), mLocalNormalPen(_localNormalPen)
	{
	}

	aos::Vec3V mLocalPointA;
	aos::Vec3V mLocalPointB;
	aos::Vec4V mLocalNormalPen; // the (x, y, z) is the local normal, and the w is the penetration depth
};

//This contact is used in the mesh contact gen to store an extra variable
class MeshPersistentContact : public PersistentContact
{
public:
	PxU32 mFaceIndex;
};

//This contact is used in the compress stream buffer(NpCacheStreamPair in the PxcNpThreadContext) to store the data we need

class CachedMeshPersistentContact
{
public:
	PxVec3 mLocalPointA; //16 byte aligned
	PxU32 mFaceIndex;	 //face index
	PxVec3 mLocalPointB; //16 byte aligned
	PxU32  mPad;		 //pad
	PxVec3 mLocalNormal; //16 byte aligned
	PxReal mPen;		 //penetration
};


//This class is used to store the start and end index of the mesh persistent contacts in an array.
struct PCMContactPatch
{
public:
	EHDI PCMContactPatch()
	{
		mNextPatch = NULL;
		mEndPatch = NULL;
		mRoot = this;
		mPatchMaxPen = aos::FMax();
	}
	aos::Vec3V mPatchNormal;
	PCMContactPatch* mNextPatch;//store the next patch pointer in the patch list
	PCMContactPatch* mEndPatch;//store the last patch pointer in the patch list
	PCMContactPatch* mRoot;//if this is the start of the patch list which has very similar patch normal, the root will be itself
	aos::FloatV mPatchMaxPen;//store the deepest penetration of the whole patch
	PxU32 mStartIndex;//store the start index of the manifold contacts in the manifold contacts stream
	PxU32 mEndIndex;//store the end index of the manifold contacts in the manifold contacts stream
	PxU32 mTotalSize;//if this is the root, the total size will store the total number of manifold contacts in the whole patch list
};


//ML: this is needed because it seems NEON doesn't force the alignment on SIMD type, which cause some of the PCM unit tests fail

class PersistentContactManifold
{
public:

	EHDI PersistentContactManifold(PersistentContact* contactPointsBuff, PxU8 capacity): mNumContacts(0), mCapacity(capacity), mNumWarmStartPoints(0), mContactPoints(contactPointsBuff)
	{
		using namespace px::aos;
		mRelativeTransform.invalidate();
		mQuatA = QuatIdentity();
		mQuatB = QuatIdentity();
	}

	EHD PxU32	getNumContacts() const { return mNumContacts;}

	EHD bool	isEmpty() const	{ return mNumContacts==0; }

	EHD PersistentContact& getContactPoint(const PxU32 index)
	{
		E_ASSERT(index < GU_MANIFOLD_CACHE_SIZE);
		return mContactPoints[index];
	}
	
	EHD aos::FloatV maxTransformPositionDelta(const aos::Vec3V& curP)	const
	{
		using namespace aos;
	
		const Vec3V deltaP = V3Sub(curP, mRelativeTransform.p);
		const Vec4V delta = Vec4V_From_Vec3V(V3Abs(deltaP));
		//need to work out max from a single vector...
		return V4ExtractMax(delta);
	}

	EHD void setRelativeTransform(const aos::PxTransformV& transform)
	{
		mRelativeTransform = transform;
	}

	EHD void setRelativeTransform(const aos::PxTransformV& transform, const aos::QuatV quatA, const aos::QuatV quatB)
	{
		mRelativeTransform = transform;
		mQuatA = quatA;
		mQuatB = quatB;
	}

	//This is used for the box/convexhull vs box/convexhull contact gen to decide whether the relative movement of a pair of objects are 
	//small enough. In this case, we can skip the collision detection all together
	EHD PxU32 invalidate_BoxConvex(const aos::PxTransformV& curRTrans, const aos::QuatV& quatA, const aos::QuatV& quatB,
		const aos::FloatVArg minMargin, const aos::FloatVArg radiusA, const aos::FloatVArg radiusB)	const
	{
		using namespace aos;
		E_ASSERT(mNumContacts <= GU_MANIFOLD_CACHE_SIZE);
		const FloatV ratio = FLoad(invalidateThresholds_p()[mNumContacts]);
		const FloatV thresholdP = FMul(minMargin, ratio);
		const FloatV deltaP = maxTransformPositionDelta(curRTrans.p);

		const FloatV thresholdQ = FLoad(invalidateQuatThresholds_p()[mNumContacts]);
		const FloatV deltaQA = QuatDot(quatA, mQuatA);
		const FloatV deltaQB = QuatDot(quatB, mQuatB);

		const BoolV con0 = BOr(FIsGrtr(deltaP, thresholdP), BOr(FIsGrtr(thresholdQ, deltaQA), FIsGrtr(thresholdQ, deltaQB)));
		PxU32 generateContacts = BAllEqTTTT(con0);
		if (!generateContacts)
		{
			PxReal dqA, dqB;
			FStore(deltaQA, &dqA);
			FStore(deltaQB, &dqB);

			const PxReal aRadian = dqA < 1.0f ? PxAcos(dqA) : 0.f;
			const FloatV travelDistA = FMul(FLoad(aRadian), radiusA);
			
			const PxReal bRadian = dqB < 1.0f ? PxAcos(dqB) : 0.f;
			const FloatV travelDistB = FMul(FLoad(bRadian), radiusB);

			const BoolV con = BOr(FIsGrtr(travelDistA, thresholdP), FIsGrtr(travelDistB, thresholdP));

			generateContacts = BAllEqTTTT(con);
		}

		return generateContacts;
	}

	//This is used for the sphere/capsule vs other primitives contact gen to decide whether the relative movement of a pair of objects are 
	//small enough. In this case, we can skip the collision detection all together
	EHD PxU32 invalidate_SphereCapsule(const aos::PxTransformV& curRTrans, const aos::FloatVArg minMargin)	const
	{
		using namespace aos;
		E_ASSERT(mNumContacts <= 2);
		const FloatV ratio = FLoad(invalidateThresholds2_p()[mNumContacts]);
		
		const FloatV thresholdP = FMul(minMargin, ratio);
		const FloatV deltaP = maxTransformPositionDelta(curRTrans.p);

		const FloatV thresholdQ = FLoad(invalidateQuatThresholds2_p()[mNumContacts]);
		const FloatV deltaQ = QuatDot(curRTrans.q, mRelativeTransform.q);
		const BoolV con = BOr(FIsGrtr(deltaP, thresholdP), FIsGrtr(thresholdQ, deltaQ));

		return BAllEqTTTT(con);
	}

	//This is used for plane contact gen to decide whether the relative movement of a pair of objects are small enough. In this case, 
	//we can skip the collision detection all together
	EHD PxU32 invalidate_PrimitivesPlane(const aos::PxTransformV& curRTrans, const aos::FloatVArg minMargin, const aos::FloatVArg ratio)	const
	{
		using namespace aos;
		const FloatV thresholdP = FMul(minMargin, ratio);
		const FloatV deltaP = maxTransformPositionDelta(curRTrans.p);
	
		const FloatV thresholdQ = FLoad(0.99996f);//about 0.5 degree
		const FloatV deltaQ = QuatDot(curRTrans.q, mRelativeTransform.q);
		const BoolV con = BOr(FIsGrtr(deltaP, thresholdP), FIsGrtr(thresholdQ, deltaQ));

		return BAllEqTTTT(con);
	}

	EHD void removeContactPoint(PxU32 index)
	{
		mNumContacts--;
		mContactPoints[index] = mContactPoints[mNumContacts];
	}      
    
	/*bool validContactDistance(const PersistentContact& pt, const aos::FloatVArg breakingThreshold) const
	{
		using namespace aos;
		const FloatV dist = V4GetW(pt.mLocalNormalPen);
		return FAllGrtr(breakingThreshold, dist) != 0;
	}*/

	EHD void clearManifold()
	{
		mNumWarmStartPoints = 0;
		mNumContacts = 0;
		mRelativeTransform.invalidate();
	}

	EHD void initialize()
	{
		clearManifold();
	}

	//This function is used to replace the existing contact with the newly created contact if their distance are within some threshold
	EHDV bool replaceManifoldPoint(const aos::Vec3VArg localPointA, const aos::Vec3VArg localPointB, const aos::Vec4VArg localNormalPen, const aos::FloatVArg replaceBreakingThreshold);

	//This function is to add a point(in box/convexhull contact gen) to the exising manifold. If the number of manifold is more than 4, we need to do contact reduction
	EHDV PxU32 addManifoldPoint(const aos::Vec3VArg localPointA, const aos::Vec3VArg localPointB, const aos::Vec4VArg localNormalAPen, const aos::FloatVArg replaceBreakingThreshold);
	//This function is to add a point(in capsule contact gen) to the exising manifold. If the number of manifold is more than 4, we need to do contact reduction
	EHDV PxU32 addManifoldPoint2(const aos::Vec3VArg localPointA, const aos::Vec3VArg localPointB, const aos::Vec4VArg localNormalAPen, const aos::FloatVArg replaceBreakingThreshold);//max two points of contacts  
	//This function is used in box-plane contact gen to add the plane contacts to the manifold 
	EHDV void addBatchManifoldContactsCluster(const PersistentContact* manifoldPoints, PxU32 numPoints);
	
	//This function is used in the capsule full manifold contact genenation(maximum 2 points). 
	EHDV void addBatchManifoldContacts2(const PersistentContact* manifoldPoints, PxU32 numPoints);//max two points of contacts             
	
	//This function is used in the box/convexhull full manifold contact generation(maximum 4 points). 
	EHDV void addBatchManifoldContacts(const PersistentContact* manifoldPoints, PxU32 numPoints, PxReal toleranceLength);
	//This function is using the cluster algorithm to reduce contacts
	EHDV void reduceBatchContactsCluster(const PersistentContact* manifoldPoints, PxU32 numPoints);
	//This function is called by addBatchManifoldContacts2 to reduce the manifold contacts to 2 points;
	EHDV void reduceBatchContacts2(const PersistentContact* manifoldPoints, PxU32 numPoints);
	//This function is called by addBatchManifoldContacts to reduce the manifold contacts to 4 points
	EHDV void reduceBatchContacts(const PersistentContact* manifoldPoints, PxU32 numPoints, PxReal toleranceLength);

	//This function is used for incremental manifold contact reduction for box/convexhull
	EHDV PxU32 reduceContactsForPCM(const aos::Vec3VArg localPointA, const aos::Vec3VArg localPointB, const aos::Vec4VArg localNormalPen);
	//This function is used for incremental manifold contact reduction for capsule
	EHDV PxU32 reduceContactSegment(const aos::Vec3VArg localPointA, const aos::Vec3VArg localPointB, const aos::Vec4VArg localNormalPen);
  
	//	This function recalculate the contacts in the manifold based on the current relative transform between a pair of objects. If the recalculated contacts are within some threshold,
	//	we will keep the contacts; Otherwise, we will remove the contacts.
	EHDV void refreshContactPoints(const aos::PxMatTransformV& relTra, const aos::FloatVArg projectBreakingThreshold, const aos::FloatVArg contactOffset);
	//This function is just used in boxbox contact gen for fast transform
	EHDV void addManifoldContactsToContactBuffer(PxContactBuffer& contactBuffer, const aos::Vec3VArg normal, const aos::PxMatTransformV& transf1);
	//This function is for adding box/convexhull manifold contacts to the contact buffer
	EHDV void addManifoldContactsToContactBuffer(PxContactBuffer& contactBuffer, const aos::Vec3VArg normal, const aos::PxTransformV& transf1, const aos::FloatVArg contactOffset);
	//This function is for adding sphere/capsule manifold contacts to the contact buffer
	EHDV void addManifoldContactsToContactBuffer(PxContactBuffer& contactBuffer, const aos::Vec3VArg normal, const aos::Vec3VArg projectionNormal, const aos::PxTransformV& transf0, const aos::FloatVArg radius, const aos::FloatVArg contactOffset);

	//get the average normal in the manifold in world space
	EHDV aos::Vec3V getWorldNormal(const aos::PxTransformV& trB)	const;
	//get the average normal in the manifold in local B object space
	EHDV aos::Vec3V getLocalNormal()	const;
	
	EHDV void recordWarmStart(PxU8* aIndices, PxU8* bIndices, PxU8& nbWarmStartPoints);
	EHDV void setWarmStart(const PxU8* aIndices, const PxU8* bIndices, const PxU8 nbWarmStartPoints);
	EHDV void drawManifold(PxRenderOutput& out, const aos::PxTransformV& trA, const aos::PxTransformV& trB)	const;
	EHDV void drawManifold(PxRenderOutput& out, const aos::PxTransformV& trA, const aos::PxTransformV& trB, const aos::FloatVArg radius)	const;
	EHDV void drawManifold(const PersistentContact& m, PxRenderOutput& out, const aos::PxTransformV& trA, const aos::PxTransformV& trB)	const;
	EHDV static void drawPoint(PxRenderOutput& out, const aos::Vec3VArg p, const PxF32 size, PxU32 color = 0x0000ff00);
	EHDV static void drawLine(PxRenderOutput& out, const aos::Vec3VArg p0, const aos::Vec3VArg p1, PxU32 color = 0xff00ffff);
	EHDV static void drawTriangle(PxRenderOutput& out, const aos::Vec3VArg p0, const aos::Vec3VArg p1, const aos::Vec3VArg p2, PxU32 color = 0xffff0000);
	EHDV static void drawTetrahedron(PxRenderOutput& out, const aos::Vec3VArg p0, const aos::Vec3VArg p1, const aos::Vec3VArg p2, const aos::Vec3VArg p3, PxU32 color = 0xffff0000);
	EHDV static void drawPolygon(PxRenderOutput& out, const aos::PxTransformV& transform, const aos::Vec3V* points, PxU32 numVerts, PxU32 color = 0xff00ffff);
	EHDV static void drawPolygon(PxRenderOutput& out, const aos::PxMatTransformV& transform, const aos::Vec3V* points, PxU32 numVerts, PxU32 color = 0xff00ffff);

	aos::PxTransformV mRelativeTransform;//aToB
	aos::QuatV mQuatA;
	aos::QuatV mQuatB;
	PxU8 mNumContacts;
	PxU8 mCapacity;
	PxU8 mNumWarmStartPoints;
	PxU8 mAIndice[4];
	PxU8 mBIndice[4]; 
	PersistentContact* mContactPoints;
} ;


class LargePersistentContactManifold : public PersistentContactManifold
{
public:
	EHDI LargePersistentContactManifold() : PersistentContactManifold(mContactPointsBuff, GU_MANIFOLD_CACHE_SIZE)
	{
	}

	PersistentContact mContactPointsBuff[GU_MANIFOLD_CACHE_SIZE];
};


class SpherePersistentContactManifold : public PersistentContactManifold
{
public:
	EHDI SpherePersistentContactManifold() : PersistentContactManifold(mContactPointsBuff, GU_SPHERE_MANIFOLD_CACHE_SIZE)
	{
	}

	PersistentContact mContactPointsBuff[GU_SPHERE_MANIFOLD_CACHE_SIZE];
};


class SinglePersistentContactManifold
{
public:

	EHDI SinglePersistentContactManifold(): mNumContacts(0)
	{
	}

	EHD PxU32	getNumContacts() const { return mNumContacts;}

	EHD bool	isEmpty()	const	{ return mNumContacts==0; }

	EHD MeshPersistentContact& getContactPoint(const PxU32 index)
	{
		E_ASSERT(index < GU_SINGLE_MANIFOLD_CACHE_SIZE);
		return mContactPoints[index];
	}

	EHD void removeContactPoint (PxU32 index)
	{
		mNumContacts--;
		mContactPoints[index] = mContactPoints[mNumContacts];
	}  

	EHD void clearManifold()
	{
		mNumContacts = 0;
	}

	EHD void initialize()
	{
		clearManifold();
	}

	EHD aos::Vec3V getWorldNormal(const aos::PxTransformV& trB)	const
	{
		using namespace aos;
		Vec4V nPen = mContactPoints[0].mLocalNormalPen;
		for(PxU32 i=1; i<mNumContacts; ++i)
		{
			nPen = V4Add(nPen, mContactPoints[i].mLocalNormalPen);
		}

		const Vec3V n = Vec3V_From_Vec4V(nPen);
		return V3Normalize(trB.rotate(n));
	}

	EHD aos::Vec3V getLocalNormal()	const
	{
		using namespace aos;
		Vec4V nPen = mContactPoints[0].mLocalNormalPen;
		for(PxU32 i=1; i<mNumContacts; ++i)
		{
			nPen = V4Add(nPen, mContactPoints[i].mLocalNormalPen);
		}
		return V3Normalize(Vec3V_From_Vec4V(nPen));
	}
	
	//This function reduces the manifold contact list in a patch for box/convexhull vs mesh
	EHDV aos::FloatV reduceBatchContactsConvex(const MeshPersistentContact* manifoldContactExt, PxU32 numContacts, PCMContactPatch& patch);
	//This function reduces the manifold contact list in a patch for sphere vs mesh
	EHDV aos::FloatV reduceBatchContactsSphere(const MeshPersistentContact* manifoldContactExt, PxU32 numContacts, PCMContactPatch& patch);
	//This function reduces the manifold contact list in a patch for capsuel vs mesh
	EHDV aos::FloatV reduceBatchContactsCapsule(const MeshPersistentContact* manifoldContactExt, PxU32 numContacts, PCMContactPatch& patch);

	//This function adds the manifold contact list in a patch for box/convexhull vs mesh
	EHDV aos::FloatV addBatchManifoldContactsConvex(const MeshPersistentContact* manifoldContact, PxU32 numContacts, PCMContactPatch& patch, const aos::FloatVArg replaceBreakingThreshold);
	//This function adds the manifold contact list in a patch for sphere vs mesh
	EHDV aos::FloatV addBatchManifoldContactsSphere(const MeshPersistentContact* manifoldContact, PxU32 numContacts, PCMContactPatch& patch, const aos::FloatVArg replaceBreakingThreshold);
	//This function adds the manifold contact list in a patch for capsule vs mesh
	EHDV aos::FloatV addBatchManifoldContactsCapsule(const MeshPersistentContact* manifoldContact, PxU32 numContacts, PCMContactPatch& patch, const aos::FloatVArg replaceBreakingThreshold);
	
	//This is used for in the addContactsToPatch for convex mesh contact gen. 
	EHDV static PxU32 reduceContacts(MeshPersistentContact* manifoldContactExt, PxU32 numContacts);

	//This function is to recalculate the contacts based on the relative transform between a pair of objects
	EHDV aos::FloatV refreshContactPoints(const aos::PxMatTransformV& relTra, const aos::FloatVArg projectBreakingThreshold, const aos::FloatVArg contactOffset);

	EHDV void drawManifold(PxRenderOutput& out, const aos::PxTransformV& trA, const aos::PxTransformV& trB)	const;

	MeshPersistentContact mContactPoints[GU_SINGLE_MANIFOLD_CACHE_SIZE];//384 bytes
	PxU32 mNumContacts;//400 bytes
	
} ;

//This is a structure used to cache a multi-persistent-manifold in the cache stream
struct MultiPersistentManifoldHeader
{
	aos::PxTransformV mRelativeTransform;//aToB
	PxU32 mNumManifolds;
	PxU32 pad[3];
};

struct SingleManifoldHeader
{
	PxU32 mNumContacts;
	PxU32 pad[3];
};



class  MultiplePersistentContactManifold
{
public:
	EHDI MultiplePersistentContactManifold():mNumManifolds(0), mNumTotalContacts(0)
	{
		mRelativeTransform.invalidate();
	}

	EHD void setRelativeTransform(const aos::PxTransformV& transform)
	{
		mRelativeTransform = transform;
	}

	EHD aos::FloatV maxTransformPositionDelta(const aos::Vec3V& curP)	const
	{
		using namespace aos;
	
		const Vec3V deltaP = V3Sub(curP, mRelativeTransform.p);
		const Vec4V delta = Vec4V_From_Vec3V(V3Abs(deltaP));
		//need to work out max from a single vector...
		return V4ExtractMax(delta);
	}

	EHD PxU32 invalidate(const aos::PxTransformV& curRTrans, const aos::FloatVArg minMargin, const aos::FloatVArg ratio)	const
	{
		using namespace aos;
		
		const FloatV thresholdP = FMul(minMargin, ratio);
		const FloatV deltaP = maxTransformPositionDelta(curRTrans.p);
		const FloatV thresholdQ = FLoad(0.9998f);//about 1 degree
		const FloatV deltaQ = QuatDot(curRTrans.q, mRelativeTransform.q);
		const BoolV con = BOr(FIsGrtr(deltaP, thresholdP), FIsGrtr(thresholdQ, deltaQ));

		return BAllEqTTTT(con);
	}

	EHD PxU32 invalidate(const aos::PxTransformV& curRTrans, const aos::FloatVArg minMargin)	const
	{
		using namespace aos;
		return invalidate(curRTrans, minMargin, FLoad(0.2f));
	}

	// This function work out the contact patch connectivity. If two patches's normal are within 5 degree, we would link these two patches together and reset the total size.
	EHD void refineContactPatchConnective(PCMContactPatch** contactPatch, PxU32 numContactPatch, MeshPersistentContact* manifoldContacts, const aos::FloatVArg acceptanceEpsilon)	const
	{
		E_UNUSED(manifoldContacts);

		using namespace aos;
	
		//work out the contact patch connectivity, the patchNormal should be in the local space of mesh
		for(PxU32 i=0; i<numContactPatch; ++i)
		{
			PCMContactPatch* patch = contactPatch[i];
			patch->mRoot = patch;
			patch->mEndPatch = patch;
			patch->mTotalSize = patch->mEndIndex - patch->mStartIndex;
			patch->mNextPatch = NULL;
	
			for(PxU32 j=i; j>0; --j)
			{
				PCMContactPatch* other = contactPatch[j-1];
				const FloatV d = V3Dot(patch->mPatchNormal, other->mRoot->mPatchNormal);
				if(FAllGrtrOrEq(d, acceptanceEpsilon))//less than 5 degree
				{
					other->mNextPatch = patch;
					other->mRoot->mEndPatch = patch;
					patch->mRoot = other->mRoot;
					other->mRoot->mTotalSize += patch->mEndIndex - patch->mStartIndex;
					break;
				}
			}
		}
	}

	// This function uses to reduce the manifold contacts which are in different connected patchs but are within replace breaking threshold 
	EHD PxU32 reduceManifoldContactsInDifferentPatches(PCMContactPatch** contactPatch, PxU32 numContactPatch, MeshPersistentContact* manifoldContacts, PxU32 numContacts, const aos::FloatVArg sqReplaceBreaking)	const
	{
		using namespace aos;

		for(PxU32 i=0; i<numContactPatch; ++i)
		{
			PCMContactPatch* currentPatch = contactPatch[i];
			//this make sure the patch is the root before we do the contact reduction, otherwise, we will do duplicate work
			if(currentPatch->mRoot == currentPatch)
			{
				while(currentPatch)
				{
					PCMContactPatch* nextPatch = currentPatch->mNextPatch;
					if(nextPatch)
					{
						for(PxU32 k = currentPatch->mStartIndex; k<currentPatch->mEndIndex; ++k)
						{
							for(PxU32 l = nextPatch->mStartIndex; l < nextPatch->mEndIndex; ++l)
							{
								Vec3V dif = V3Sub(manifoldContacts[l].mLocalPointB, manifoldContacts[k].mLocalPointB);
								FloatV d = V3Dot(dif, dif);
								if(FAllGrtr(sqReplaceBreaking, d))
								{
									//if two manifold contacts are within threshold, we will get rid of the manifold contacts in the other contact patch
									manifoldContacts[l] = manifoldContacts[nextPatch->mEndIndex-1];
									nextPatch->mEndIndex--;
									numContacts--;
									l--;
								}
							}
						}
					}
					currentPatch = nextPatch;
				}
			}
		}
	
		return numContacts;
	}

	// This function is for the multiple manifold loop through each individual single manifold to recalculate the contacts based on the relative transform between a pair of objects
	EHD void refreshManifold(const aos::PxMatTransformV& relTra, const aos::FloatVArg projectBreakingThreshold, const aos::FloatVArg contactDist)
	{
		using namespace aos;
	
		//refresh manifold contacts
		for(PxU32 i=0; i<mNumManifolds; ++i)
		{
			const PxU8 ind = mManifoldIndices[i];
			E_ASSERT(mManifoldIndices[i] < GU_MAX_MANIFOLD_SIZE);
			const PxU32 nextInd = PxMin(i, mNumManifolds-2u)+1;
			PxPrefetchLine(&mManifolds[mManifoldIndices[nextInd]]);
			PxPrefetchLine(&mManifolds[mManifoldIndices[nextInd]],128);
			PxPrefetchLine(&mManifolds[mManifoldIndices[nextInd]],256);
			const FloatV _maxPen = mManifolds[ind].refreshContactPoints(relTra, projectBreakingThreshold, contactDist);
			if(mManifolds[ind].isEmpty())
			{
				//swap the index with the next manifolds
				const PxU8 index = mManifoldIndices[--mNumManifolds];
				mManifoldIndices[mNumManifolds] = ind;
				mManifoldIndices[i] = index;
				i--;
			}
			else
			{
				FStore(_maxPen, &mMaxPen[ind]);
			}
		}
	}

	EHD void initialize()
	{
		mNumManifolds = 0;
		mNumTotalContacts = 0;
		mRelativeTransform.invalidate();
		for(PxU8 i=0; i<GU_MAX_MANIFOLD_SIZE; ++i)
		{
			mManifolds[i].initialize();
			mManifoldIndices[i] = i;
		}
	}

	EHD void clearManifold()
	{
		for(PxU8 i=0; i<mNumManifolds; ++i)
		{
			mManifolds[i].clearManifold();
		}
		mNumManifolds = 0;
		mNumTotalContacts = 0;
		mRelativeTransform.invalidate();
	}

	EHD const SinglePersistentContactManifold* getManifold(PxU32 index)	const
	{
		E_ASSERT(index < GU_MAX_MANIFOLD_SIZE);
		return &mManifolds[mManifoldIndices[index]];
	}

	EHD SinglePersistentContactManifold* getManifold(PxU32 index)
	{
		E_ASSERT(index < GU_MAX_MANIFOLD_SIZE);
		return &mManifolds[mManifoldIndices[index]];
	}

	EHD SinglePersistentContactManifold* getEmptyManifold()
	{
		if(mNumManifolds < GU_MAX_MANIFOLD_SIZE)
			return  &mManifolds[mManifoldIndices[mNumManifolds]];
		return NULL;
	}

	//This function adds the manifold contacts with different patches into the corresponding single persistent contact manifold
	EHDV void addManifoldContactPoints(MeshPersistentContact* manifoldContact, PxU32 numManifoldContacts, PCMContactPatch** contactPatch, PxU32 numPatch, 
		const aos::FloatVArg sqReplaceBreakingThreshold, const aos::FloatVArg acceptanceEpsilon, PxU8 maxContactsPerManifold);
	//This function adds the box/convexhull manifold contacts to the contact buffer 
	EHDV bool addManifoldContactsToContactBuffer(PxContactBuffer& contactBuffer, const aos::PxTransformV& transf1);
	//This function adds the sphere/capsule manifold contacts to the contact buffer
	EHDV bool addManifoldContactsToContactBuffer(PxContactBuffer& contactBuffer, const aos::PxTransformV& trA, const aos::PxTransformV& trB, const aos::FloatVArg radius);
	EHDV void drawManifold(PxRenderOutput& out, const aos::PxTransformV& trA, const aos::PxTransformV& trB)	const;

	//Code to load from a buffer and store to a buffer.
	EHDV void fromBuffer(PxU8*  buffer);
	EHDV void toBuffer(PxU8*  buffer)	const;

	EHDV static void drawLine(PxRenderOutput& out, const aos::Vec3VArg p0, const aos::Vec3VArg p1, PxU32 color = 0xff00ffff);
	EHDV static void drawLine(PxRenderOutput& out, const PxVec3 p0, const PxVec3 p1, PxU32 color = 0xff00ffff);
	EHDV static void drawPoint(PxRenderOutput& out, const aos::Vec3VArg p, const PxF32 size, PxU32 color = 0x00ff0000);
	EHDV static void drawPolygon(PxRenderOutput& out, const aos::PxTransformV& transform, aos::Vec3V* points, PxU32 numVerts, PxU32 color = 0xff00ffff);

	aos::PxTransformV mRelativeTransform;//aToB
	PxF32 mMaxPen[GU_MAX_MANIFOLD_SIZE];
	PxU8 mManifoldIndices[GU_MAX_MANIFOLD_SIZE];
	PxU8 mNumManifolds;
	PxU8 mNumTotalContacts;
	SinglePersistentContactManifold mManifolds[GU_MAX_MANIFOLD_SIZE];
	
} ;


// This function calculates the average normal in the manifold in world space
EHD aos::Vec3V PersistentContactManifold::getWorldNormal(const aos::PxTransformV& trB)	const
{
	using namespace aos;
	
	Vec4V nPen = mContactPoints[0].mLocalNormalPen;
	for(PxU32 i=1; i<mNumContacts; ++i)
		nPen = V4Add(nPen, mContactPoints[i].mLocalNormalPen);

	const Vec3V n = Vec3V_From_Vec4V(nPen);
	const FloatV sqLength = V3Dot(n, n);
	const Vec3V nn = V3Sel(FIsGrtr(sqLength, FEps()), n, Vec3V_From_Vec4V(mContactPoints[0].mLocalNormalPen));
	return trB.rotateAndNormalize(nn);
}

// This function calculates the average normal in the manifold in local B space
EHD aos::Vec3V PersistentContactManifold::getLocalNormal()	const
{
	using namespace aos;
	
	Vec4V nPen = mContactPoints[0].mLocalNormalPen;
	for(PxU32 i=1; i<mNumContacts; ++i)
		nPen = V4Add(nPen, mContactPoints[i].mLocalNormalPen);

	return V3Normalize(Vec3V_From_Vec4V(nPen));
}

// This function recalculates the contacts in the manifold based on the current relative transform between a pair of objects. If the recalculated contacts are within some threshold,
// we will keep the contacts; Otherwise, we will remove the contacts.
EHD void PersistentContactManifold::refreshContactPoints(const aos::PxMatTransformV& aToB, const aos::FloatVArg projectBreakingThreshold, const aos::FloatVArg /*contactOffset*/)
{
	using namespace aos;
	const FloatV sqProjectBreakingThreshold = FMul(projectBreakingThreshold, projectBreakingThreshold); 

	// first refresh worldspace positions and distance
	for (PxU32 i=mNumContacts; i > 0; --i)
	{
		PersistentContact& manifoldPoint = mContactPoints[i-1];
		const Vec3V localAInB = aToB.transform( manifoldPoint.mLocalPointA ); // from a to b
		const Vec3V localBInB = manifoldPoint.mLocalPointB;
		const Vec3V v = V3Sub(localAInB, localBInB); 

		const Vec3V localNormal = Vec3V_From_Vec4V(manifoldPoint.mLocalNormalPen); // normal in b space
		const FloatV dist = V3Dot(v, localNormal);

		const Vec3V projectedPoint = V3NegScaleSub(localNormal, dist, localAInB);//manifoldPoint.worldPointA - manifoldPoint.worldPointB * manifoldPoint.m_distance1;
		const Vec3V projectedDifference = V3Sub(localBInB, projectedPoint);

		const FloatV distance2d = V3Dot(projectedDifference, projectedDifference);
		//const BoolV con = BOr(FIsGrtr(dist, contactOffset), FIsGrtr(distance2d, sqProjectBreakingThreshold));
		const BoolV con = FIsGrtr(distance2d, sqProjectBreakingThreshold);
		if(BAllEqTTTT(con))
		{
			removeContactPoint(i-1);
		} 
		else
		{
			manifoldPoint.mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), dist);
		}
	}
}

#define EPX_CP_TO_PCP(contactPoint)	(reinterpret_cast<PersistentContact*>(contactPoint)) //this is used in the normal pcm contact gen
#define EPX_CP_TO_MPCP(contactPoint)	(reinterpret_cast<MeshPersistentContact*>(contactPoint))//this is used in the mesh pcm contact gen

EHDV void addManifoldPoint(PersistentContact* manifoldContacts, PersistentContactManifold& manifold, GjkOutput& output,
	const aos::PxMatTransformV& aToB, const aos::FloatV replaceBreakingThreshold);

}//Gu
}//physx

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactGenUtil.h
//The smallest epsilon we will permit (scaled by PxTolerancesScale.length)
#define	PCM_WITNESS_POINT_LOWER_EPS		1e-2f
//The largest epsilon we will permit (scaled by PxTolerancesScale.length)
#define	PCM_WITNESS_POINT_UPPER_EPS		5e-2f

namespace px
{
namespace Gu
{
	enum FeatureStatus
	{
		POLYDATA0,
		POLYDATA1,
		EDGE
	};

	EHDV bool contains(aos::Vec3V* verts, PxU32 numVerts, const aos::Vec3VArg p, const aos::Vec3VArg min, const aos::Vec3VArg max);

	EHD aos::FloatV signed2DTriArea(const aos::Vec3VArg a, const aos::Vec3VArg b, const aos::Vec3VArg c)
	{
		using namespace aos;
		const Vec3V ca = V3Sub(a, c);
		const Vec3V cb = V3Sub(b, c);

		const FloatV t0 = FMul(V3GetX(ca), V3GetY(cb));
		const FloatV t1 = FMul(V3GetY(ca), V3GetX(cb));

		return FSub(t0, t1);
	}

	EHDV PxI32 getPolygonIndex(const Gu::PolygonalData& polyData, const SupportLocal* map, const aos::Vec3VArg normal, PxI32& polyIndex2);

	EHD PxI32 getPolygonIndex(const Gu::PolygonalData& polyData, const SupportLocal* map, const aos::Vec3VArg normal)
	{
		using namespace aos;

		PxI32 index2;
		return getPolygonIndex(polyData, map, normal, index2);
	}

	EHDV PxU32 getWitnessPolygonIndex(	const Gu::PolygonalData& polyData, const SupportLocal* map, const aos::Vec3VArg normal,
									const aos::Vec3VArg closest, PxReal tolerance);

	EHD void outputPCMContact(PxContactBuffer& contactBuffer, PxU32& contactCount, const aos::Vec3VArg point, const aos::Vec3VArg normal,
											const aos::FloatVArg penetration, PxU32 internalFaceIndex1 = PXC_CONTACT_NO_FACE_INDEX)
	{
		using namespace aos;

		// PT: TODO: the PCM capsule-capsule code was using this alternative version, is it better?
		// const Vec4V normalSep = V4SetW(Vec4V_From_Vec3V(normal), separation);
		// V4StoreA(normalSep, &point.normal.x);
		// Also aren't we overwriting maxImpulse with the position now? Ok to do so?
		E_ASSERT(contactCount < PxContactBuffer::MAX_CONTACTS);
		PxContactPoint& contact = contactBuffer.contacts[contactCount++];
		V4StoreA(Vec4V_From_Vec3V(normal), &contact.normal.x);
		V4StoreA(Vec4V_From_Vec3V(point), &contact.point.x);
		FStore(penetration, &contact.separation);
		E_ASSERT(contact.point.isFinite());
		E_ASSERT(contact.normal.isFinite());
		E_ASSERT(PxIsFinite(contact.separation));
		contact.internalFaceIndex1 = internalFaceIndex1;
	}

	EHD bool outputSimplePCMContact(PxContactBuffer& contactBuffer, const aos::Vec3VArg point, const aos::Vec3VArg normal,
												const aos::FloatVArg penetration, PxU32 internalFaceIndex1 = PXC_CONTACT_NO_FACE_INDEX)
	{
		outputPCMContact(contactBuffer, contactBuffer.count, point, normal, penetration, internalFaceIndex1);
		return true;
	}

}//Gu
}//physx

// ===== 원본: physx/source/geomutils/src/pcm/GuPersistentContactManifold.cpp
using namespace px;
using namespace aos;
using namespace Gu;

//	This local function is to avoid DLL call
EHDI static FloatV distancePointSegmentSquaredLocal(const Vec3VArg a, const Vec3VArg b, const Vec3VArg p)
{
	const FloatV zero = FZero();
	const FloatV one = FOne();

	const Vec3V ap = V3Sub(p, a);
	const Vec3V ab = V3Sub(b, a);
	const FloatV nom = V3Dot(ap, ab);
	
	const FloatV denom = V3Dot(ab, ab);
	const FloatV tValue = FClamp(FDiv(nom, denom), zero, one);

	const FloatV t = FSel(FIsEq(denom, zero), zero, tValue);
	const Vec3V v = V3NegScaleSub(ab, t, ap);
	return V3Dot(v, v);
}

//	This local function is to avoid DLL call
EHDI static FloatV distancePointTriangleSquaredLocal(const Vec3VArg p, 
												const Vec3VArg a, 
												const Vec3VArg b, 
												const Vec3VArg c)
{
	const FloatV zero = FZero();
	//const Vec3V zero = V3Zero();
	const Vec3V ab = V3Sub(b, a);
	const Vec3V ac = V3Sub(c, a);
	const Vec3V bc = V3Sub(c, b);
	const Vec3V ap = V3Sub(p, a);
	const Vec3V bp = V3Sub(p, b);
	const Vec3V cp = V3Sub(p, c);

	const FloatV d1 = V3Dot(ab, ap); //  snom
	const FloatV d2 = V3Dot(ac, ap); //  tnom
	const FloatV d3 = V3Dot(ab, bp); // -sdenom
	const FloatV d4 = V3Dot(ac, bp); //  unom = d4 - d3
	const FloatV d5 = V3Dot(ab, cp); //  udenom = d5 - d6
	const FloatV d6 = V3Dot(ac, cp); // -tdenom
	const FloatV unom = FSub(d4, d3);
	const FloatV udenom = FSub(d5, d6);
	
	//check if p in vertex region outside a
	const BoolV con00 = FIsGrtr(zero, d1); // snom <= 0
	const BoolV con01 = FIsGrtr(zero, d2); // tnom <= 0
	const BoolV con0 = BAnd(con00, con01); // vertex region a

	if(BAllEqTTTT(con0))
	{
		const Vec3V vv = V3Sub(p, a);
		return V3Dot(vv, vv);
	}

	//check if p in vertex region outside b
	const BoolV con10 = FIsGrtrOrEq(d3, zero);
	const BoolV con11 = FIsGrtrOrEq(d3, d4);
	const BoolV con1 = BAnd(con10, con11); // vertex region b
	if(BAllEqTTTT(con1))
	{
		const Vec3V vv = V3Sub(p, b);
		return V3Dot(vv, vv);
	}

	//check if p in vertex region outside c
	const BoolV con20 = FIsGrtrOrEq(d6, zero);
	const BoolV con21 = FIsGrtrOrEq(d6, d5); 
	const BoolV con2 = BAnd(con20, con21); // vertex region c
	if(BAllEqTTTT(con2))
	{
		const Vec3V vv = V3Sub(p, c);
		return V3Dot(vv, vv);
	}

	//check if p in edge region of AB
	const FloatV vc = FSub(FMul(d1, d4), FMul(d3, d2));
	
	const BoolV con30 = FIsGrtr(zero, vc);
	const BoolV con31 = FIsGrtrOrEq(d1, zero);
	const BoolV con32 = FIsGrtr(zero, d3);
	const BoolV con3 = BAnd(con30, BAnd(con31, con32));
	if(BAllEqTTTT(con3))
	{
		const FloatV sScale = FDiv(d1, FSub(d1, d3));
		const Vec3V closest3 = V3ScaleAdd(ab, sScale, a);//V3Add(a, V3Scale(ab, sScale));
		const Vec3V vv = V3Sub(p, closest3);
		return V3Dot(vv, vv);
	}

	//check if p in edge region of BC
	const FloatV va = FSub(FMul(d3, d6),FMul(d5, d4));
	const BoolV con40 = FIsGrtr(zero, va);
	const BoolV con41 = FIsGrtrOrEq(d4, d3);
	const BoolV con42 = FIsGrtrOrEq(d5, d6);
	const BoolV con4 = BAnd(con40, BAnd(con41, con42)); 
	if(BAllEqTTTT(con4))
	{
		const FloatV uScale = FDiv(unom, FAdd(unom, udenom));
		const Vec3V closest4 = V3ScaleAdd(bc, uScale, b);//V3Add(b, V3Scale(bc, uScale));
		const Vec3V vv = V3Sub(p, closest4);
		return V3Dot(vv, vv);
	}

	//check if p in edge region of AC
	const FloatV vb = FSub(FMul(d5, d2), FMul(d1, d6));
	const BoolV con50 = FIsGrtr(zero, vb);
	const BoolV con51 = FIsGrtrOrEq(d2, zero);
	const BoolV con52 = FIsGrtr(zero, d6);
	const BoolV con5 = BAnd(con50, BAnd(con51, con52));
	if(BAllEqTTTT(con5))
	{
		const FloatV tScale = FDiv(d2, FSub(d2, d6));
		const Vec3V closest5 = V3ScaleAdd(ac, tScale, a);//V3Add(a, V3Scale(ac, tScale));
		const Vec3V vv = V3Sub(p, closest5);
		return V3Dot(vv, vv);
	}

	//P must project inside face region. Compute Q using Barycentric coordinates
	const Vec3V n = V3Cross(ab, ac);
	const FloatV nn = V3Dot(n,n);
	const FloatV t = FSel(FIsGrtr(nn, zero),FDiv(V3Dot(n, V3Sub(a, p)), nn), zero);
	const Vec3V closest6 = V3Add(p, V3Scale(n, t));

	const Vec3V vv = V3Sub(p, closest6);

	return V3Dot(vv, vv);
}

namespace px
{
namespace Gu
{
//This is the translational threshold used in invalidate_BoxConvexHull. 0.5 is 50% of the object margin. we use different threshold between
//0 and 4 points. This threashold is a scale that is multiplied by the objects' margins.
E_GTABLE(PxF32, invalidateThresholds, 5, { 0.5f, 0.125f, 0.25f, 0.375f, 0.375f })

//This is the translational threshold used in invalidate_SphereCapsule. 0.5 is 50% of the object margin, we use different threshold between
//0 and 2 points. This threshold is a scale that is multiplied by the objects' margin
E_GTABLE(PxF32, invalidateThresholds2, 3, { 0.5f, 0.1f, 0.75f })

//This is the rotational threshold used in invalidate_BoxConvexHull. 0.9998 is a threshold for quat difference
//between previous and current frame
E_GTABLE(PxF32, invalidateQuatThresholds, 5, { 0.9998f, 0.9999f, 0.9999f, 0.9999f, 0.9999f })

//This is the rotational threshold used in invalidate_SphereCapsule. 0.9995f is a threshold for quat difference
//between previous and current frame
E_GTABLE(PxF32, invalidateQuatThresholds2, 3, { 0.9995f, 0.9999f, 0.9997f })
}
}


EHDI static void drawManifoldPoint(const PersistentContact& manifold, const PxTransformV& trA, const PxTransformV& trB, PxRenderOutput& out, PxU32 color=0xffffff)
{
	E_UNUSED(color);

	const Vec3V worldA = trA.transform(manifold.mLocalPointA);
	const Vec3V worldB = trB.transform(manifold.mLocalPointB);
	const Vec3V localNormal = Vec3V_From_Vec4V(manifold.mLocalNormalPen);
	const FloatV pen = V4GetW(manifold.mLocalNormalPen);
	
	const Vec3V worldNormal = trB.rotate(localNormal);
	PxVec3 a, b, v;
	V3StoreU(worldA, a);
	V3StoreU(worldB, b);
	V3StoreU(worldNormal, v);
	PxF32 dist;
	FStore(pen, &dist);
	PxVec3 e = a - v*dist;

	PxF32 size = 0.05f;
	const PxVec3 up(0.f, size, 0.f);
	const PxVec3 right(size, 0.f, 0.f);
	const PxVec3 forwards(0.f, 0.f, size);

	PxF32 size2 = 0.1f;
	const PxVec3 up2(0.f, size2, 0.f);
	const PxVec3 right2(size2, 0.f, 0.f);
	const PxVec3 forwards2(0.f, 0.f, size2);
	
	const PxMat44 m(PxIdentity);
	
	out << m << PxRenderOutput::LINES;

	out << 0xffff00ff << a << e;
	out << 0xff00ffff << a + up << a - up;
	out << 0xff00ffff << a + right << a - right;
	out << 0xff00ffff << a + forwards << a - forwards;

	out << 0xffff0000 << b + up2 << b - up2;
	out << 0xffff0000 << b + right2 << b - right2;
	out << 0xffff0000 << b + forwards2 << b - forwards2;

	out << 0xffff0000 << a << b;

	const PxVec3 c = a - v*10.f;
	out << 0xffff00ff << a << c;
}

EHDI static void drawManifoldPoint(const PersistentContact& manifold, const PxTransformV& trA, const PxTransformV& trB, const FloatVArg radius, PxRenderOutput& out, PxU32 color=0xffffff)
{
	E_UNUSED(color);

	const Vec3V localNormal = Vec3V_From_Vec4V(manifold.mLocalNormalPen);
	const Vec3V worldNormal = trB.rotate(localNormal);
	const Vec3V worldA = V3NegScaleSub(worldNormal, radius, trA.transform(manifold.mLocalPointA));
	const Vec3V worldB = trB.transform(manifold.mLocalPointB);
	const FloatV pen = FSub(V4GetW(manifold.mLocalNormalPen), radius);
	
	PxVec3 a, b, v;
	V3StoreU(worldA, a);
	V3StoreU(worldB, b);
	V3StoreU(worldNormal, v);
	PxF32 dist;
	FStore(pen, &dist);
	PxVec3 e = a - v*dist;

	PxF32 size = 0.05f;
	const PxVec3 up(0.f, size, 0.f);
	const PxVec3 right(size, 0.f, 0.f);
	const PxVec3 forwards(0.f, 0.f, size);

	PxF32 size2 = 0.1f;
	const PxVec3 up2(0.f, size2, 0.f);
	const PxVec3 right2(size2, 0.f, 0.f);
	const PxVec3 forwards2(0.f, 0.f, size2);
	
	const PxMat44 m(PxIdentity);
	
	out << 0xffff00ff << m << PxRenderOutput::LINES  << a << e;
	out << 0xff00ffff << m << PxRenderOutput::LINES << a + up << a - up;
	out << 0xff00ffff << m << PxRenderOutput::LINES << a + right << a - right;
	out << 0xff00ffff << m << PxRenderOutput::LINES << a + forwards << a - forwards;

	out << 0xffff0000 << m << PxRenderOutput::LINES << b + up2 << b - up2;
	out << 0xffff0000 << m << PxRenderOutput::LINES << b + right2 << b - right2;
	out << 0xffff0000 << m << PxRenderOutput::LINES << b + forwards2 << b - forwards2;

	out << 0xffff0000 << m << PxRenderOutput::LINES << a << b;
}

static PxU32 gColors[8] = { 0xff0000ff, 0xff00ff00, 0xffff0000,
							0xff00ffff, 0xffff00ff, 0xffffff00,
							0xff000080, 0xff008000};

//	SIMD version 
EHDI Mat33V Gu::findRotationMatrixFromZAxis(const Vec3VArg to)
{
	const FloatV one = FOne();
	const FloatV threshold = FLoad(0.9999f);

	const FloatV e = V3GetZ(to);
	const FloatV f = FAbs(e);

	if(FAllGrtr(threshold, f))
	{
		const FloatV vx = FNeg(V3GetY(to));
		const FloatV vy = V3GetX(to);
		const FloatV h = FRecip(FAdd(one, e)); 
		const FloatV hvx = FMul(h,vx);
		const FloatV hvxy = FMul(hvx, vy);

		const Vec3V col0 = V3Merge(FScaleAdd(hvx, vx, e),	hvxy,							vy);
		const Vec3V col1 = V3Merge(hvxy,					FScaleAdd(h, FMul(vy, vy), e),	FNeg(vx));
		const Vec3V col2 = V3Merge(FNeg(vy),				vx,								e);

		return Mat33V(col0, col1, col2);
	}
	else
	{
		const FloatV two = FLoad(2.f);
		const Vec3V from = V3UnitZ();
		const Vec3V absFrom = V3UnitY();

		const Vec3V u = V3Sub(absFrom, from);
		const Vec3V v = V3Sub(absFrom, to);

		const FloatV dotU = V3Dot(u, u);
		const FloatV dotV = V3Dot(v, v);
		const FloatV dotUV = V3Dot(u, v);

		const FloatV c1 = FNeg(FDiv(two, dotU));
		const FloatV c2 = FNeg(FDiv(two, dotV));
		const FloatV c3 = FMul(c1, FMul(c2, dotUV));

		const Vec3V c1u = V3Scale(u, c1);
		const Vec3V c2v = V3Scale(v, c2);
		const Vec3V c3v = V3Scale(v, c3);

		FloatV temp0 = V3GetX(c1u);
		FloatV temp1 = V3GetX(c2v);
		FloatV temp2 = V3GetX(c3v);

		Vec3V col0 = V3ScaleAdd(u, temp0, V3ScaleAdd(v, temp1, V3Scale(u, temp2)));
		col0 = V3SetX(col0, FAdd(V3GetX(col0), one));

		temp0 = V3GetY(c1u);
		temp1 = V3GetY(c2v);
		temp2 = V3GetY(c3v);

		Vec3V col1 = V3ScaleAdd(u, temp0, V3ScaleAdd(v, temp1, V3Scale(u, temp2)));
		col1 = V3SetY(col1, FAdd(V3GetY(col1), one));

		temp0 = V3GetZ(c1u);
		temp1 = V3GetZ(c2v);
		temp2 = V3GetZ(c3v);

		Vec3V col2 = V3ScaleAdd(u, temp0, V3ScaleAdd(v, temp1, V3Scale(u, temp2)));
		col2 = V3SetZ(col2, FAdd(V3GetZ(col2), one));

		return Mat33V(col0, col1, col2);
	}
}

EHDI void PersistentContactManifold::drawManifold(PxRenderOutput& out, const PxTransformV& trA, const PxTransformV& trB) const
{
	PxVec3 a, b;
	V3StoreU(trA.p, a);
	V3StoreU(trB.p, b);

	for(PxU32 i = 0; i< mNumContacts; ++i)
	{
		PersistentContact& m = mContactPoints[i];
		drawManifoldPoint(m, trA, trB, out, gColors[i]);
	}
}

EHDI void PersistentContactManifold::drawManifold(PxRenderOutput& out, const PxTransformV& trA, const PxTransformV& trB, const FloatVArg radius) const
{
	PxVec3 a, b;
	V3StoreU(trA.p, a);
	V3StoreU(trB.p, b);

	for(PxU32 i = 0; i< mNumContacts; ++i)
	{
		PersistentContact& m = mContactPoints[i];
		drawManifoldPoint(m, trA, trB, radius, out, gColors[i]);
	}
}

EHDI void PersistentContactManifold::drawManifold(const PersistentContact& m, PxRenderOutput& out, const PxTransformV& trA, const PxTransformV& trB) const
{	
	drawManifoldPoint(m, trA, trB, out, gColors[0]);
}

EHDI void PersistentContactManifold::drawPoint(PxRenderOutput& out, const Vec3VArg p, const PxF32 size, PxU32 color)
{
	const PxVec3 up(0.f, size, 0.f);
	const PxVec3 right(size, 0.f, 0.f);
	const PxVec3 forwards(0.f, 0.f, size);

	PxVec3 a;
	V3StoreU(p, a);

	const PxMat44 m(PxIdentity);
	
	out << color << m << PxRenderOutput::LINES << a + up << a - up;
	out << color << m << PxRenderOutput::LINES << a + right << a - right;
	out << color << m << PxRenderOutput::LINES << a + forwards << a - forwards;
}

EHDI void PersistentContactManifold::drawLine(PxRenderOutput& out, const Vec3VArg p0, const Vec3VArg p1, PxU32 color)
{
	PxVec3 a, b;
	V3StoreU(p0, a);
	V3StoreU(p1, b);

	const PxMat44 m(PxIdentity);
	out << color << m << PxRenderOutput::LINES << a << b;
}

EHDI void PersistentContactManifold::drawTriangle(PxRenderOutput& out, const Vec3VArg p0, const Vec3VArg p1, const Vec3VArg p2, PxU32 color)
{
	PxVec3 a, b, c;
	V3StoreU(p0, a);
	V3StoreU(p1, b);
	V3StoreU(p2, c);

	const PxMat44 m(PxIdentity);
	out << color << m << PxRenderOutput::TRIANGLES << a << b << c;
}

EHDI void PersistentContactManifold::drawTetrahedron(PxRenderOutput& out, const Vec3VArg p0, const Vec3VArg p1, const Vec3VArg p2, const Vec3VArg p3, PxU32 color)
{
	PxVec3 a, b, c, d;
	V3StoreU(p0, a);
	V3StoreU(p1, b);
	V3StoreU(p2, c);
	V3StoreU(p3, d);

	const PxMat44 m(PxIdentity);
	out << color << m << PxRenderOutput::LINES << a << b;
	out << color << m << PxRenderOutput::LINES << a << c;
	out << color << m << PxRenderOutput::LINES << a << d;
	out << color << m << PxRenderOutput::LINES << b << c;
	out << color << m << PxRenderOutput::LINES << b << d;
	out << color << m << PxRenderOutput::LINES << c << d;
}

EHDI void PersistentContactManifold::drawPolygon(PxRenderOutput& out, const PxTransformV& transform, const Vec3V* points, PxU32 numVerts, PxU32 color)
{
	for(PxU32 i=0; i<numVerts; ++i)
	{
		Vec3V tempV0 = points[i == 0 ? numVerts-1 : i-1];
		Vec3V tempV1 = points[i];
		
		drawLine(out, transform.transform(tempV0), transform.transform(tempV1), color);
	}
}

EHDI void PersistentContactManifold::drawPolygon(PxRenderOutput& out, const PxMatTransformV& transform, const Vec3V* points, PxU32 numVerts, PxU32 color)
{
	for(PxU32 i=0; i<numVerts; ++i)
	{
		Vec3V tempV0 = points[i == 0 ? numVerts-1 : i-1];
		Vec3V tempV1 = points[i];
		
		drawLine(out, transform.transform(tempV0), transform.transform(tempV1), color);
	}
}

EHDI void PersistentContactManifold::recordWarmStart(PxU8* aIndices, PxU8* bIndices, PxU8& nbWarmStartPoints)
{
	nbWarmStartPoints = mNumWarmStartPoints;
	for(PxU8 i = 0; i < mNumWarmStartPoints; ++i)
	{
		aIndices[i] = mAIndice[i];
		bIndices[i] = mBIndice[i];
	}
}

EHDI void PersistentContactManifold::setWarmStart(const PxU8* aIndices, const PxU8* bIndices, const PxU8 nbWarmStartPoints)
{
	mNumWarmStartPoints = nbWarmStartPoints;
	for(PxU8 i = 0; i < nbWarmStartPoints; ++i)
	{
		mAIndice[i] = aIndices[i];
		mBIndice[i] = bIndices[i];
	}
}

//	If a new point and the exisitng point's distance are within some replace breaking threshold, we will replace the existing point with the new point.
//	This is used for incremental manifold strategy.
EHDI bool PersistentContactManifold::replaceManifoldPoint(const Vec3VArg localPointA, const Vec3VArg localPointB, const Vec4VArg localNormalPen
														, const FloatVArg replaceBreakingThreshold)
{
	const FloatV shortestDist = FMul(replaceBreakingThreshold, replaceBreakingThreshold); 
	
	for(PxU32 i=0; i<mNumContacts; ++i)
	{
		const PersistentContact& mp = mContactPoints[i];

		const Vec3V diffB = V3Sub(mp.mLocalPointB, localPointB);
		const FloatV sqDifB = V3Dot(diffB, diffB);
		const Vec3V diffA = V3Sub(mp.mLocalPointA, localPointA);
		const FloatV sqDifA = V3Dot(diffA, diffA);
		const FloatV minSqDif = FMin(sqDifB, sqDifA);

		if(FAllGrtr(shortestDist, minSqDif))
		{
			mContactPoints[i].mLocalPointA = localPointA;
			mContactPoints[i].mLocalPointB = localPointB;
			mContactPoints[i].mLocalNormalPen = localNormalPen;
			return true;
		}
	}
	return false;
}

EHDI PxU32 PersistentContactManifold::reduceContactSegment(const Vec3VArg localPointA, const Vec3VArg localPointB, const Vec4VArg localNormalPen)
{
	const Vec3V p  = localPointB;
	const Vec3V p0 = mContactPoints[0].mLocalPointB;
	const Vec3V p1 = mContactPoints[1].mLocalPointB;
	const Vec3V v0 = V3Sub(p0, p);
	const Vec3V v1 = V3Sub(p1, p);
	const FloatV dist0 = V3Dot(v0, v0);
	const FloatV dist1 = V3Dot(v1, v1);
	if(FAllGrtr(dist0, dist1))
	{
		mContactPoints[1].mLocalPointA = localPointA;
		mContactPoints[1].mLocalPointB = localPointB;
		mContactPoints[1].mLocalNormalPen = localNormalPen;
	}
	else
	{
		mContactPoints[0].mLocalPointA = localPointA;
		mContactPoints[0].mLocalPointB = localPointB;
		mContactPoints[0].mLocalNormalPen = localNormalPen;
	}

	return 0;
}

EHDI PxU32 PersistentContactManifold::reduceContactsForPCM(const Vec3VArg localPointA, const Vec3VArg localPointB, const Vec4VArg localNormalPen)
{
	bool chosen[5];
	PxMemZero(chosen, sizeof(bool)*5);
	const FloatV negMax = FNeg(FMax());
	PersistentContact tempContacts[5];
	
	for(PxU32 i=0; i<4; ++i)
	{
		tempContacts[i] = mContactPoints[i];
	}
	tempContacts[4].mLocalPointA = localPointA;
	tempContacts[4].mLocalPointB = localPointB;
	tempContacts[4].mLocalNormalPen = localNormalPen;

	//ML: we set the start point to be the 4th point
	FloatV maxDist = V4GetW(localNormalPen);
	PxI32 index = 4;
	//Choose deepest point
	for(PxI32 i=0; i<4; ++i)
	{
		const FloatV pen = V4GetW(tempContacts[i].mLocalNormalPen);
		if(FAllGrtr(maxDist, pen))
		{
			maxDist = pen;
			index = i;
		}
	}

	chosen[index] = true;
	mContactPoints[0] = tempContacts[index];

	//ML: we set the start point to be the 0th point 
	Vec3V dir = V3Sub(tempContacts[0].mLocalPointB, mContactPoints[0].mLocalPointB);
	maxDist = V3Dot(dir, dir);
	index = 0;

	for(PxI32 i=1; i<5; ++i)
	{
		if(!chosen[i])
		{
			dir = V3Sub(tempContacts[i].mLocalPointB, mContactPoints[0].mLocalPointB);
			const FloatV d = V3Dot(dir, dir);
			if(FAllGrtr(d, maxDist))
			{
				maxDist = d;
				index = i;
			}
		}
	}

	//E_ASSERT(chosen[index] == false);
	chosen[index] = true;
	mContactPoints[1] = tempContacts[index];

	maxDist = negMax;	
	for(PxI32 i=0; i<5; ++i)
	{
		if(!chosen[i])
		{
			const FloatV sqDif = distancePointSegmentSquaredLocal(mContactPoints[0].mLocalPointB, mContactPoints[1].mLocalPointB, tempContacts[i].mLocalPointB);
			if(FAllGrtr(sqDif, maxDist))
			{
				maxDist = sqDif;
				index = i;
			}
		}
	}
	//E_ASSERT(chosen[index] == false);
	chosen[index] = true;
	mContactPoints[2]=tempContacts[index];

	//Find point farthest away from segment tempContactPoints[0] - tempContactPoints[1]
	maxDist = negMax;
	for(PxI32 i=0; i<5; ++i)
	{
		if(!chosen[i])
		{
			const FloatV sqDif = distancePointTriangleSquaredLocal(	tempContacts[i].mLocalPointB, mContactPoints[0].mLocalPointB, mContactPoints[1].mLocalPointB, mContactPoints[2].mLocalPointB); 
			if(FAllGrtr(sqDif, maxDist))
			{
				maxDist = sqDif;
				index = i;
			}
		}
	}

	//E_ASSERT(chosen[index] == false);
	if(chosen[index] == true)
	{
		//if we don't have any new contacts, which means the leftover contacts are inside the triangles
		mNumContacts = 3;
		return 0;
	}
	else
	{
		chosen[index] = true;
		mContactPoints[3] = tempContacts[index];
	}

	//Final pass, we work out the index that we didn't choose and bind it to its closest point. We then consider whether we want to swap the point if the
	//point we were about to discard is deeper...

	PxU32 notChosenIndex = 0;
	for(PxU32 a = 0; a < 5; ++a)
	{
		if(!chosen[a])
		{
			notChosenIndex = a;
			break;
		}
	}

	FloatV closest = FMax();
	index = 0;
	for(PxI32 a = 0; a < 4; ++a)
	{
		Vec3V dif = V3Sub(mContactPoints[a].mLocalPointA, tempContacts[notChosenIndex].mLocalPointA);
		const FloatV d2 = V3Dot(dif, dif);
		if(FAllGrtr(closest, d2))
		{
			closest = d2;
			index = a;
		}
	}

	if(FAllGrtr(V4GetW(mContactPoints[index].mLocalNormalPen), V4GetW(tempContacts[notChosenIndex].mLocalNormalPen)))
	{
		//Swap
		mContactPoints[index] = tempContacts[notChosenIndex];
	}

	return 0;
}

//	This function is for box/convexHull vs box/convexHull.
EHDI void PersistentContactManifold::addManifoldContactsToContactBuffer(PxContactBuffer& contactBuffer, const Vec3VArg normal, const PxTransformV& transf1, const FloatVArg contactOffset)
{
	//add the manifold contacts;
	PxU32 contactCount = 0;
	for(PxU32 i=0; (i< mNumContacts) & (contactCount < PxContactBuffer::MAX_CONTACTS); ++i)
	{
		const PersistentContact& p = getContactPoint(i);
		
		const FloatV dist = V4GetW(p.mLocalNormalPen);

		//Either the newly created points or the cache points might have dist/penetration larger than contactOffset
		if(FAllGrtrOrEq(contactOffset, dist))
		{
			const Vec3V worldP = transf1.transform(p.mLocalPointB);

			outputPCMContact(contactBuffer, contactCount, worldP, normal, dist);
		}
	}

	contactBuffer.count = contactCount;
}

//	This function is for direct implementation for box vs box. We don't need to discard the contacts based on whether the contact offset is larger than penetration/dist because
//	the direct implementation will guarantee the penetration/dist won't be larger than the contact offset. Also, we won't have points from the cache if the direct implementation
//	of box vs box is called.
EHDI void PersistentContactManifold::addManifoldContactsToContactBuffer(PxContactBuffer& contactBuffer, const Vec3VArg normal, const PxMatTransformV& transf1)
{
	//add the manifold contacts;
	PxU32 contactCount = 0;
	for(PxU32 i=0; (i< mNumContacts) & (contactCount < PxContactBuffer::MAX_CONTACTS); ++i)
	{
		const PersistentContact& p = getContactPoint(i);
		
		const Vec3V worldP = transf1.transform(p.mLocalPointB);
		const FloatV dist = V4GetW(p.mLocalNormalPen);

		outputPCMContact(contactBuffer, contactCount, worldP, normal, dist);
	}

	contactBuffer.count = contactCount;
}

//	This function is for sphere/capsule vs other primitives. We treat sphere as a point and capsule as a segment in the contact gen and store the sphere center or a point in the segment for capsule
//	in the manifold. 
EHDI void PersistentContactManifold::addManifoldContactsToContactBuffer(PxContactBuffer& contactBuffer, const Vec3VArg normal, const Vec3VArg projectionNormal,
	const PxTransformV& transf0, const FloatVArg radius, const FloatVArg contactOffset)
{
	//add the manifold contacts;
	PxU32 contactCount = 0;
	for(PxU32 i=0; (i< mNumContacts) & (contactCount < PxContactBuffer::MAX_CONTACTS); ++i)
	{
		const PersistentContact& p = getContactPoint(i);
		const FloatV dist = FSub(V4GetW(p.mLocalNormalPen), radius);

		//The newly created points should have a dist < contactOffset. However, points might come from the PCM contact cache so we might still have points which are
		//larger than contactOffset. The reason why we don't want to discard the contacts in the contact cache whose dist > contactOffset is because the contacts may 
		//only temporarily become separated. The points still project onto roughly the same spot but so, if the bodies move together again, the contacts may be valid again.
		//This is important when simulating at large time-steps because GJK can only generate 1 point of contact and missing contacts for just a single frame with large time-steps
		//may introduce noticeable instability.
		if(FAllGrtrOrEq(contactOffset, dist))
		{
			const Vec3V worldP = V3NegScaleSub(projectionNormal, radius, transf0.transform(p.mLocalPointA));

			outputPCMContact(contactBuffer, contactCount, worldP, normal, dist);
		}
	}

	contactBuffer.count = contactCount;
}

//	This function is used in the box/convexhull full manifold contact genenation. We will pass in a list of manifold contacts. If the number of contacts are more than
//	GU_MANIFOLD_CACHE_SIZE, we will need to do contact reduction while we are storing the chosen manifold contacts from the manifold contact list to the manifold contact
//	buffer.
EHDI void PersistentContactManifold::addBatchManifoldContacts(const PersistentContact* manifoldContacts, PxU32 numPoints, PxReal toleranceLength)
{
	if(numPoints <= GU_MANIFOLD_CACHE_SIZE)
	{
		for(PxU32 i=0; i<numPoints; ++i)
		{
			mContactPoints[i].mLocalPointA = manifoldContacts[i].mLocalPointA;
			mContactPoints[i].mLocalPointB = manifoldContacts[i].mLocalPointB;
			mContactPoints[i].mLocalNormalPen = manifoldContacts[i].mLocalNormalPen;
		}
		mNumContacts = PxTo8(numPoints);
	}
	else
	{
		reduceBatchContacts(manifoldContacts, numPoints, toleranceLength);

		mNumContacts = GU_MANIFOLD_CACHE_SIZE;
	}
}  

//	This function is for the plane and box contact gen. If the number of points passed in is more than 4, we need to do contact reduction 
EHDI void PersistentContactManifold::addBatchManifoldContactsCluster(const PersistentContact* manifoldContacts, PxU32 numPoints)
{
	if(numPoints <= GU_MANIFOLD_CACHE_SIZE)
	{
		for(PxU32 i=0; i<numPoints; ++i)
		{
			mContactPoints[i].mLocalPointA = manifoldContacts[i].mLocalPointA;
			mContactPoints[i].mLocalPointB = manifoldContacts[i].mLocalPointB;
			mContactPoints[i].mLocalNormalPen = manifoldContacts[i].mLocalNormalPen;
		}
		mNumContacts = PxTo8(numPoints);
	}
	else
	{
		reduceBatchContactsCluster(manifoldContacts, numPoints);
		mNumContacts = GU_MANIFOLD_CACHE_SIZE;
	}
}

//	This function is called by addBatchManifoldContactCluster. The logic in this funtion is:
//	(1)get the depest-penetrated contacts and store it in mContactPoints[0]
//	(2)get the furthest away point from mContactPoints[0] and store in mContactPoints[1]
//	(3)calculate the min and max distance point away the segment (mContactPoints[0] and mContactPoints[1]) and store the max distance point to mContactPoints[2]
//	(4)if the min and max distance on the same side of the segment, we need to chose the min distance point again and store this point to mContactPoints[3]; 
//	(5)cluster around that 4 points and chose the deepest points
//	(6)reassign contact points
EHDI void PersistentContactManifold::reduceBatchContactsCluster(const PersistentContact* manifoldPoints, PxU32 numPoints)
{
	//get the deepest points

	bool chosen[64];
	PxMemZero(chosen, sizeof(bool)*numPoints);
	const FloatV max = FMax();
	const FloatV nmax = FNeg(max);
	FloatV maxDist = max;
	PxU32 index = 0;

	PxU32 indices[4];
	
	//get the deepest point
	for(PxU32 i=0; i<numPoints; ++i)
	{
		const FloatV dist = V4GetW(manifoldPoints[i].mLocalNormalPen);
		if(FAllGrtr(maxDist, dist))
		{
			maxDist = dist;
			index = i;
		}
	}

	//keep the furthest points in the first position
	mContactPoints[0] = manifoldPoints[index];
	chosen[index] = true;
	indices[0] = index;

	//calculate the furthest away points from mContactPoints[0]
	Vec3V v = V3Sub(manifoldPoints[0].mLocalPointB, mContactPoints[0].mLocalPointB);
	maxDist = V3Dot(v, v);
	index = 0;

	for(PxU32 i=1; i<numPoints; ++i)
	{
		v = V3Sub(manifoldPoints[i].mLocalPointB, mContactPoints[0].mLocalPointB);
		const FloatV d = V3Dot(v, v);
		if(FAllGrtr(d, maxDist))
		{
			maxDist = d;
			index = i;
		}
	}

	//E_ASSERT(chosen[index] == false);
	mContactPoints[1] = manifoldPoints[index];
	chosen[index] = true;
	indices[1] = index;

	maxDist = nmax;
	index = GU_MANIFOLD_INVALID_INDEX;
	v = V3Sub(mContactPoints[1].mLocalPointB, mContactPoints[0].mLocalPointB);

	const Vec3V cn0 = Vec3V_From_Vec4V(mContactPoints[0].mLocalNormalPen);
	Vec3V norm = V3Cross(v, cn0);
	const FloatV sqLen = V3Dot(norm, norm);
	norm = V3Sel(FIsGrtr(sqLen, FZero()), V3ScaleInv(norm, FSqrt(sqLen)), cn0);

	FloatV minDist = max;
	PxU32 index1 = GU_MANIFOLD_INVALID_INDEX;
	
	//calculate the min and max point away from the segment
	for(PxU32 i=0; i<numPoints; ++i)
	{
		if(!chosen[i])
		{
			v = V3Sub(manifoldPoints[i].mLocalPointB, mContactPoints[0].mLocalPointB);
			const FloatV d = V3Dot(v, norm);
			if(FAllGrtr(d, maxDist))
			{
				maxDist = d;
				index = i;
			}

			if(FAllGrtr(minDist, d))
			{
				minDist = d;
				index1 = i;
			}
		}
	}

	//E_ASSERT(chosen[index] == false && chosen[index1] == false);
	mContactPoints[2] = manifoldPoints[index];
	chosen[index] = true;
	indices[2] = index;

	//if min and max in the same side, chose again
	const FloatV temp = FMul(minDist, maxDist);
	if(FAllGrtr(temp, FZero()))
	{
		//chose again
		maxDist = nmax;
		for(PxU32 i=0; i<numPoints; ++i)
		{
			if(!chosen[i])
			{
				v = V3Sub(manifoldPoints[i].mLocalPointB, mContactPoints[0].mLocalPointB);
				const FloatV d = V3Dot(v, norm);
				if(FAllGrtr(d, maxDist))
				{
					maxDist = d;
					index1 = i;
				}
			}
		}
	}

	mContactPoints[3] = manifoldPoints[index1];
	chosen[index1] = true;
	indices[3] = index1;

	//cluster around that 4 points and chose the deepest points
	for(PxU32 i=0; i<numPoints; ++i)
	{
		if(!chosen[i])
		{
			maxDist = max;
			const FloatV pen = V4GetW(manifoldPoints[i].mLocalNormalPen);
			index = 0;
			for(PxU32 j=0; j<4; ++j)
			{
				const Vec3V v1 = V3Sub(manifoldPoints[i].mLocalPointB, mContactPoints[j].mLocalPointB);
				const FloatV dist = V3Dot(v1, v1);
				if(FAllGrtr(maxDist, dist))
				{
					maxDist = dist;
					index = j;
				}
			}

			//check to see whether the penetration is deeper than the point in mContactPoints
			const FloatV tempPen = V4GetW(manifoldPoints[indices[index]].mLocalNormalPen);
			if(FAllGrtr(tempPen, pen))
			{
				//swap the indices
				indices[index] = i;
			}
		}
	}

	mContactPoints[0] = manifoldPoints[indices[0]];
	mContactPoints[1] = manifoldPoints[indices[1]];
	mContactPoints[2] = manifoldPoints[indices[2]];
	mContactPoints[3] = manifoldPoints[indices[3]];				
}

//	This function is for box/convexhull full contact generation. If the numPoints > 4, we will reduce the contact points to 4
EHDI void PersistentContactManifold::reduceBatchContacts(const PersistentContact* manifoldPoints, PxU32 numPoints, PxReal tolereanceLength)
{
	PxU8 chosenIndices[4];
	PxU8 candidates[64];
	
	const FloatV zero = FZero();
	const FloatV max = FMax();
	const FloatV nmax = FNeg(max);
	FloatV maxPen = V4GetW(manifoldPoints[0].mLocalNormalPen);
	FloatV minPen = maxPen;
	PxU32 index = 0;
	candidates[0] = 0;
	PxU32 candidateIndex = 0;

	PxU32 nbCandiates = numPoints;
	//keep the deepest point, candidateIndex will be the same as index
	for(PxU32 i = 1; i<numPoints; ++i)
	{
		//at the begining candidates and indices will be the same
		candidates[i] = PxU8(i);
	
		const FloatV pen = V4GetW(manifoldPoints[i].mLocalNormalPen);
		minPen = FMax(minPen, pen);
		if(FAllGrtr(maxPen, pen))
		{
			maxPen = pen;
			index = i;
			candidateIndex = i;
		}
	}

	//keep the deepest points in the first position
	chosenIndices[0] = PxU8(index);

	//move the chosen indices out of the candidates indices
	nbCandiates = nbCandiates - 1;
	candidates[candidateIndex] = candidates[nbCandiates];
	//indices[index] = nbCandiates;

	//calculate the furthest away points
	Vec3V v = V3Sub(manifoldPoints[candidates[0]].mLocalPointB, manifoldPoints[chosenIndices[0]].mLocalPointB);
	FloatV maxDist = V3Dot(v, v);
	index = candidates[0];
	candidateIndex = 0;

	for(PxU32 i = 1; i<nbCandiates; ++i)
	{
		v = V3Sub(manifoldPoints[candidates[i]].mLocalPointB, manifoldPoints[chosenIndices[0]].mLocalPointB);
		const FloatV d = V3Dot(v, v);
		if(FAllGrtr(d, maxDist))
		{
			maxDist = d;
			index = candidates[i];
			candidateIndex = i;
		}
	}

	chosenIndices[1] = PxU8(index);
	//move the chosen indices out of the candidates indices
	nbCandiates = nbCandiates - 1;
	candidates[candidateIndex] = candidates[nbCandiates];

	v = V3Sub(manifoldPoints[chosenIndices[1]].mLocalPointB, manifoldPoints[chosenIndices[0]].mLocalPointB);
	const Vec3V cn0 = Vec3V_From_Vec4V(manifoldPoints[chosenIndices[0]].mLocalNormalPen);
	Vec3V norm = V3Cross(v, cn0);
	const FloatV sqLen = V3Dot(norm, norm);
	norm = V3Sel(FIsGrtr(sqLen, zero), V3ScaleInv(norm, FSqrt(sqLen)), cn0);

	//reset maxDist and index
	maxDist = nmax;
	index = GU_MANIFOLD_INVALID_INDEX;
	candidateIndex = GU_MANIFOLD_INVALID_INDEX;
	FloatV minDist = max;
	PxU32 index1 = GU_MANIFOLD_INVALID_INDEX;
	PxU32 candidateIndex1 = GU_MANIFOLD_INVALID_INDEX;

	//calculate the min and max point away from the segment
	for(PxU32 i = 0; i<nbCandiates; ++i)
	{
		v = V3Sub(manifoldPoints[candidates[i]].mLocalPointB, manifoldPoints[chosenIndices[0]].mLocalPointB);
		const FloatV d = V3Dot(v, norm);
		if(FAllGrtr(d, maxDist))
		{
			maxDist = d;
			index = candidates[i];
			candidateIndex = i;
		}

		if(FAllGrtr(minDist, d))
		{
			minDist = d;
			index1 = candidates[i];
			candidateIndex1 = i;
		}
	}

	chosenIndices[2] = PxU8(index);
	//move the chosen indices out of the candidates indices
	nbCandiates = nbCandiates - 1;
	candidates[candidateIndex] = candidates[nbCandiates];
	if(nbCandiates == candidateIndex1)
		candidateIndex1 = candidateIndex;

	//if min and max in the same side, chose again
	const FloatV temp = FMul(minDist, maxDist);
	if(FAllGrtr(temp, zero))
	{
		//chose again
		maxDist = nmax;
		for(PxU32 i = 0; i < nbCandiates; ++i)
		{
			v = V3Sub(manifoldPoints[candidates[i]].mLocalPointB, manifoldPoints[chosenIndices[0]].mLocalPointB);
			const FloatV d = V3Dot(v, norm);
			if(FAllGrtr(d, maxDist))
			{
				maxDist = d;
				index1 = candidates[i];
				candidateIndex1 = i;
			}
		}
	}

	chosenIndices[3] = PxU8(index1);
	//move the chosen indices out of the candidates indices
	nbCandiates = nbCandiates - 1;
	candidates[candidateIndex1] = candidates[nbCandiates];

	const FloatV eps = FLoad(tolereanceLength * 0.02f);
	const BoolV con = BAnd(FIsGrtr(eps, maxPen), FIsGrtr(minPen, eps));
	if(BAllEqTTTT(con))
	{
		//post process 
		for(PxU32 i = 0; i < 4; ++i)
		{
			FloatV pen = V4GetW(manifoldPoints[chosenIndices[i]].mLocalNormalPen);

			if(FAllGrtr(pen, eps))
			{
				candidateIndex = GU_MANIFOLD_INVALID_INDEX;
				for(PxU32 j = 0; j < nbCandiates; ++j)
				{
					const FloatV pen1 = V4GetW(manifoldPoints[candidates[j]].mLocalNormalPen);
					if(FAllGrtr(pen, pen1) && FAllGrtr(eps, pen1))
					{
						pen = pen1;
						candidateIndex = j;
					}
				}

				if(candidateIndex < nbCandiates)
				{
					const PxU8 originalIndex = chosenIndices[i];
					chosenIndices[i] = candidates[candidateIndex];
					candidates[candidateIndex] = originalIndex;
				}
			}
			mContactPoints[i] = manifoldPoints[chosenIndices[i]];
		}
	}
	else
	{
		for(PxU32 i = 0; i < 4; ++i)
		{
			mContactPoints[i] = manifoldPoints[chosenIndices[i]];
		}
	}
}

//	This function is for capsule full contact generation. If the numPoints > 2, we will reduce the contact points to 2
EHDI void PersistentContactManifold::reduceBatchContacts2(const PersistentContact* manifoldPoints, PxU32 numPoints)
{
	E_ASSERT(numPoints < 64);
	bool chosen[64];
	PxMemZero(chosen, sizeof(bool)*numPoints);
	FloatV maxDis = V4GetW(manifoldPoints[0].mLocalNormalPen);
	PxI32 index = 0;
	//keep the deepest point
	for(PxU32 i=1; i<numPoints; ++i)
	{
		const FloatV pen = V4GetW(manifoldPoints[i].mLocalNormalPen);
		if(FAllGrtr(maxDis, pen))
		{  
			maxDis = pen;
			index = PxI32(i);
		}
	}
	//keep the deepest points in the first position
	mContactPoints[0] = manifoldPoints[index];
	chosen[index] = true;

	//calculate the furthest away points
	Vec3V v = V3Sub(manifoldPoints[0].mLocalPointB, mContactPoints[0].mLocalPointB);
	maxDis = V3Dot(v, v);
	index = 0;

	for(PxU32 i=1; i<numPoints; ++i)
	{
		v = V3Sub(manifoldPoints[i].mLocalPointB, mContactPoints[0].mLocalPointB);
		const FloatV d = V3Dot(v, v);
		if(FAllGrtr(d, maxDis))
		{
			maxDis = d;
			index = PxI32(i);
		}
	}

	//E_ASSERT(chosen[index] == false);
	mContactPoints[1] = manifoldPoints[index];
	chosen[index] = true;

	PxI32 secondIndex = index;
	FloatV maxDepth = V4GetW(manifoldPoints[index].mLocalNormalPen);
	for(PxU32 i = 0; i < numPoints; ++i)
	{
		if(!chosen[i])
		{
			const Vec3V d0 = V3Sub(mContactPoints[0].mLocalPointB, manifoldPoints[i].mLocalPointB);
			const Vec3V d1 = V3Sub(mContactPoints[1].mLocalPointB, manifoldPoints[i].mLocalPointB);
			const FloatV dd0 = V3Dot(d0, d0);
			const FloatV dd1 = V3Dot(d1, d1);

			if(FAllGrtr(dd0, dd1))
			{
				//This clusters to point 1
				if(FAllGrtr(maxDepth, V4GetW(manifoldPoints[i].mLocalNormalPen)))
				{
					secondIndex = PxI32(i);
				}
			}
		}
	}

	if(secondIndex != index)
	{
		mContactPoints[1] = manifoldPoints[secondIndex];
	}	
}

EHDI PxU32 PersistentContactManifold::addManifoldPoint(const Vec3VArg localPointA, const Vec3VArg localPointB, const Vec4VArg localNormalPen, const FloatVArg replaceBreakingThreshold)
{
	if(replaceManifoldPoint(localPointA, localPointB, localNormalPen, replaceBreakingThreshold)) //replace the new point with the old one
		return 0;

	switch(mNumContacts)
	{
	case 0:
	case 1:
	case 2:
	case 3:
		mContactPoints[mNumContacts].mLocalPointA = localPointA;
		mContactPoints[mNumContacts].mLocalPointB = localPointB;
		mContactPoints[mNumContacts++].mLocalNormalPen = localNormalPen;

		return 1;
	default:
		return reduceContactsForPCM(localPointA, localPointB, localNormalPen);//should be always return zero
	};
}

//	This function is for capsule vs other primitives. If the manifold originally has contacts and we can incrementally add a point at a time, we will
//	use this function to add a point to manifold. If the number of contacts inside the manifold is more than 2, we will reduce contacts to 2 points.
EHDI PxU32 PersistentContactManifold::addManifoldPoint2(const Vec3VArg localPointA, const Vec3VArg localPointB, const Vec4VArg localNormalPen, const FloatVArg replaceBreakingThreshold)
{
	if(replaceManifoldPoint(localPointA, localPointB, localNormalPen, replaceBreakingThreshold)) //replace the new point with the old one
		return 0;

	switch(mNumContacts)
	{
	case 0:
	case 1:      
		mContactPoints[mNumContacts].mLocalPointA = localPointA;
		mContactPoints[mNumContacts].mLocalPointB = localPointB;
		mContactPoints[mNumContacts++].mLocalNormalPen = localNormalPen;
		return 1;
	case 2:
		return reduceContactSegment(localPointA, localPointB, localNormalPen);
	default:
		E_ASSERT(0);
	};
	return 0;
}

//	This function is used in the capsule full manifold contact genenation. We will pass in a list of manifold contacts. If the number of contacts are more than
//	2, we will need to do contact reduction while we are storing the chosen manifold contacts from the manifold contact list to the manifold contact buffer.
EHDI void PersistentContactManifold::addBatchManifoldContacts2(const PersistentContact* manifoldContacts, PxU32 numPoints)
{
	if(numPoints <= 2)
	{
		for(PxU32 i=0; i<numPoints; ++i)
		{
			mContactPoints[i].mLocalPointA = manifoldContacts[i].mLocalPointA;
			mContactPoints[i].mLocalPointB = manifoldContacts[i].mLocalPointB;
			mContactPoints[i].mLocalNormalPen = manifoldContacts[i].mLocalNormalPen;
		}

		mNumContacts = PxTo8(numPoints);
	}
	else
	{
		reduceBatchContacts2(manifoldContacts, numPoints);
		mNumContacts = 2;
	}
}


//	If the patch total number of manifold contacts are less than or equal to GU_SINGLE_MANIFOLD_CACHE_SIZE, we will add the manifold contacts in the contact stream to the manifold contact buffer
//	which is associated to this single persistent contact manifold. Otherwise, we will reduce the manifold contacts to GU_SINGLE_MANIFOLD_CACHE_SIZE.
EHDI FloatV SinglePersistentContactManifold::addBatchManifoldContactsConvex(const MeshPersistentContact* manifoldContact, PxU32 numContactExt, PCMContactPatch& patch, const FloatVArg replaceBreakingThreshold)
{
	E_UNUSED(replaceBreakingThreshold);

	if(patch.mTotalSize <= GU_SINGLE_MANIFOLD_CACHE_SIZE)
	{
		PCMContactPatch* currentPatch = &patch;

		//this is because we already add the manifold contacts into manifoldContact array
		PxU32 tempNumContacts = 0;
		while(currentPatch)
		{
			for(PxU32 j=currentPatch->mStartIndex; j<currentPatch->mEndIndex; ++j)
			{
				mContactPoints[tempNumContacts++] = manifoldContact[j];
			}
			currentPatch = currentPatch->mNextPatch;
		} 
		mNumContacts = tempNumContacts;
		return patch.mPatchMaxPen;
	}
	else
	{
		//contact reduction
		const FloatV maxPen = reduceBatchContactsConvex(manifoldContact, numContactExt, patch);
		mNumContacts = GU_SINGLE_MANIFOLD_CACHE_SIZE;
		return maxPen;
	}
}

EHDI FloatV SinglePersistentContactManifold::addBatchManifoldContactsSphere(const MeshPersistentContact* manifoldContact, PxU32 numContactExt, PCMContactPatch& patch, const FloatVArg replaceBreakingThreshold)
{
	E_UNUSED(replaceBreakingThreshold);

	const FloatV maxPen = reduceBatchContactsSphere(manifoldContact, numContactExt, patch);
	mNumContacts = 1;
	return maxPen;
}

EHDI FloatV SinglePersistentContactManifold::addBatchManifoldContactsCapsule(const MeshPersistentContact* manifoldContact, PxU32 numContactExt, PCMContactPatch& patch, const FloatVArg replaceBreakingThreshold)
{
	E_UNUSED(replaceBreakingThreshold);

	if(patch.mTotalSize <=GU_CAPSULE_MANIFOLD_CACHE_SIZE)
	{
		PCMContactPatch* currentPatch = &patch;

		//this is because we already add the manifold contacts into manifoldContact array
		PxU32 tempNumContacts = 0;
		while(currentPatch)
		{
			for(PxU32 j=currentPatch->mStartIndex; j<currentPatch->mEndIndex; ++j)
			{
				mContactPoints[tempNumContacts++] = manifoldContact[j];
			}
			currentPatch = currentPatch->mNextPatch;
		} 
		mNumContacts = tempNumContacts;
		return patch.mPatchMaxPen;
	}
	else
	{
		const FloatV maxPen = reduceBatchContactsCapsule(manifoldContact, numContactExt, patch);
		mNumContacts = GU_CAPSULE_MANIFOLD_CACHE_SIZE;
		return maxPen;
	}
}

EHDI FloatV SinglePersistentContactManifold::reduceBatchContactsSphere(const MeshPersistentContact* manifoldContactExt, PxU32 numContacts, PCMContactPatch& patch)
{
	E_UNUSED(numContacts);

	FloatV max = FMax();
	FloatV maxDist = max;
	PxI32 index = -1;

	PCMContactPatch* currentPatch = &patch;
	while(currentPatch)
	{
		for(PxU32 i=currentPatch->mStartIndex; i<currentPatch->mEndIndex; ++i)
		{
			const FloatV pen = V4GetW(manifoldContactExt[i].mLocalNormalPen);
			if(FAllGrtr(maxDist, pen))
			{
				maxDist = pen;
				index = PxI32(i);
			}
		}
		currentPatch = currentPatch->mNextPatch;
	}

	E_ASSERT(index!=-1);
	mContactPoints[0] = manifoldContactExt[index];

	return maxDist;
}

EHDI FloatV SinglePersistentContactManifold::reduceBatchContactsCapsule(const MeshPersistentContact* manifoldContactExt, PxU32 numContacts, PCMContactPatch& patch)
{
	bool chosen_buf[E_ALLOCA_N]; bool* chosen = chosen_buf;
	PxMemZero(chosen, sizeof(bool)*numContacts);
	const FloatV max = FMax();
	FloatV maxDis = max;
	PxI32 index = -1;

	FloatV maxPen = max;

	PCMContactPatch* currentPatch = &patch;
	while(currentPatch)
	{
		for(PxU32 i=currentPatch->mStartIndex; i<currentPatch->mEndIndex; ++i)
		{
			const FloatV pen = V4GetW(manifoldContactExt[i].mLocalNormalPen);
			if(FAllGrtr(maxDis, pen))
			{
				maxDis = pen;
				index = PxI32(i);
			}
		}
		currentPatch = currentPatch->mNextPatch;
	}

	chosen[index] = true;
	mContactPoints[0] = manifoldContactExt[index];
	maxPen = FMin(maxPen, V4GetW(manifoldContactExt[index].mLocalNormalPen));

	//calculate the furthest away points
	Vec3V v = V3Sub(manifoldContactExt[patch.mStartIndex].mLocalPointB, mContactPoints[0].mLocalPointB);
	maxDis = V3Dot(v, v);
	index = PxI32(patch.mStartIndex);

	currentPatch = &patch;
	while(currentPatch)
	{
		for(PxU32 i=currentPatch->mStartIndex; i<currentPatch->mEndIndex; ++i)
		{
			v = V3Sub(manifoldContactExt[i].mLocalPointB, mContactPoints[0].mLocalPointB);
			const FloatV d = V3Dot(v, v);
			if(FAllGrtr(d, maxDis))
			{
				maxDis = d;
				index = PxI32(i);
			}
		}
		currentPatch = currentPatch->mNextPatch;
	}

	//E_ASSERT(chosen[index] == false);
	chosen[index] = true;
	mContactPoints[1] = manifoldContactExt[index];
	maxPen = FMin(maxPen, V4GetW(manifoldContactExt[index].mLocalNormalPen));

	//keep the second deepest point
	maxDis = max;
	currentPatch = &patch;
	while(currentPatch)
	{
		for(PxU32 i=currentPatch->mStartIndex; i<currentPatch->mEndIndex; ++i)
		{
			if(!chosen[i])
			{
				const FloatV pen = V4GetW(manifoldContactExt[i].mLocalNormalPen);
				//const FloatV v = V3Dot(manifoldContactExt[i].mLocalPointB, manifoldContactExt[i].mLocalPointB);
				if(FAllGrtr(maxDis, pen))
				{
					maxDis = pen;
					index = PxI32(i);
				}
			}
		}
		currentPatch = currentPatch->mNextPatch;
	}
	//E_ASSERT(chosen[index] == false);
	chosen[index] = true;
	mContactPoints[2] = manifoldContactExt[index];
	maxPen = FMin(maxPen, V4GetW(manifoldContactExt[index].mLocalNormalPen));

	return maxPen;
}

EHDI FloatV SinglePersistentContactManifold::reduceBatchContactsConvex(const MeshPersistentContact* manifoldContactExt, PxU32 numContacts, PCMContactPatch& patch)
{
	bool chosen_buf[E_ALLOCA_N]; bool* chosen = chosen_buf;
	PxMemZero(chosen, sizeof(bool)*numContacts);
	const FloatV max = FMax();
	const FloatV nmax = FNeg(max);
	FloatV maxDis = nmax;
	PxU32 index = GU_MANIFOLD_INVALID_INDEX;

	PCMContactPatch* currentPatch = &patch;
	while(currentPatch)
	{
		for(PxU32 i=currentPatch->mStartIndex; i<currentPatch->mEndIndex; ++i)
		{
			//const FloatV pen = V4GetW(manifoldContactExt[i].localNormalPen);
			const FloatV v = V3Dot(manifoldContactExt[i].mLocalPointB, manifoldContactExt[i].mLocalPointB);
			if(FAllGrtr(v, maxDis))
			{
				maxDis = v;
				index = i;
			}
		}
		currentPatch = currentPatch->mNextPatch;
	}

	chosen[index] = true;

	const Vec3V contact0 = manifoldContactExt[index].mLocalPointB;
	mContactPoints[0] = manifoldContactExt[index];

	FloatV maxPen = V4GetW(manifoldContactExt[index].mLocalNormalPen);

	//calculate the furthest away points
	Vec3V v = V3Sub(manifoldContactExt[patch.mStartIndex].mLocalPointB, contact0);
	maxDis = V3Dot(v, v);
	index = patch.mStartIndex;

	currentPatch = &patch;
	while(currentPatch)
	{
		for(PxU32 i=currentPatch->mStartIndex; i<currentPatch->mEndIndex; ++i)
		{
			v = V3Sub(manifoldContactExt[i].mLocalPointB, contact0);
			const FloatV d = V3Dot(v, v);
			if(FAllGrtr(d, maxDis))
			{
				maxDis = d;
				index = i;
			}
		}
		currentPatch = currentPatch->mNextPatch;
	}

	//E_ASSERT(chosen[index] == false);
	chosen[index] = true;
	const Vec3V contact1 = manifoldContactExt[index].mLocalPointB;
	mContactPoints[1] = manifoldContactExt[index];

	maxPen = FMin(maxPen, V4GetW(manifoldContactExt[index].mLocalNormalPen));

	maxDis = nmax;
	index = GU_MANIFOLD_INVALID_INDEX;
	v = V3Sub(contact1, contact0);
	const Vec3V cn0 = Vec3V_From_Vec4V(mContactPoints[0].mLocalNormalPen);
	Vec3V norm = V3Cross(v, cn0);
	const FloatV sqLen = V3Dot(norm, norm);
	norm = V3Sel(FIsGrtr(sqLen, FZero()), V3ScaleInv(norm, FSqrt(sqLen)), cn0);
	FloatV minDis = max;
	PxU32 index1 = GU_MANIFOLD_INVALID_INDEX;
	
	//calculate the point furthest way to the segment
	currentPatch = &patch;
	
	while(currentPatch)
	{
		for(PxU32 i=currentPatch->mStartIndex; i<currentPatch->mEndIndex; ++i)
		{
			if(!chosen[i])
			{
				v = V3Sub(manifoldContactExt[i].mLocalPointB, contact0);
				const FloatV d = V3Dot(v, norm);
				if(FAllGrtr(d, maxDis))
				{
					maxDis = d;
					index = i;
				}

				if(FAllGrtr(minDis, d))
				{
					minDis = d;
					index1 = i;
				}
			}
		}
		currentPatch = currentPatch->mNextPatch;
	}

	//E_ASSERT(chosen[index] == false);

	chosen[index] = true;
	mContactPoints[2] = manifoldContactExt[index];
	maxPen = FMin(maxPen, V4GetW(manifoldContactExt[index].mLocalNormalPen));

	//if min and max in the same side, choose again
	const FloatV temp = FMul(minDis, maxDis);
	if(FAllGrtr(temp, FZero()))
	{
		//choose again
		maxDis = nmax;
		//calculate the point furthest way to the segment
		currentPatch = &patch;
		
		while(currentPatch)
		{
			for(PxU32 i=currentPatch->mStartIndex; i<currentPatch->mEndIndex; ++i)
			{
				if(!chosen[i])
				{
					v = V3Sub(manifoldContactExt[i].mLocalPointB, contact0);
					const FloatV d = V3Dot(v, norm);
					if(FAllGrtr(d, maxDis))
					{
						maxDis = d;
						index1 = i;
					}
				}
			}
			currentPatch = currentPatch->mNextPatch;
		}
	}

	//E_ASSERT(chosen[index1] == false);

	chosen[index1] = true;
	mContactPoints[3] = manifoldContactExt[index1];
	maxPen = FMin(maxPen, V4GetW(manifoldContactExt[index1].mLocalNormalPen));

	const PxU32 NB_TO_ADD = GU_SINGLE_MANIFOLD_CACHE_SIZE - 4;

	FloatV pens[NB_TO_ADD];
	PxU32 inds[NB_TO_ADD];
	for(PxU32 a = 0; a < NB_TO_ADD; ++a)
	{
		pens[a] = FMax();
		inds[a] = 0;
	}
	{
		currentPatch = &patch;

		while(currentPatch)
		{
			for(PxU32 i = currentPatch->mStartIndex; i < currentPatch->mEndIndex; ++i)
			{
				if(!chosen[i])
				{
					const FloatV pen = V4GetW(manifoldContactExt[i].mLocalNormalPen);
					for(PxU32 a = 0; a < NB_TO_ADD; ++a)
					{
						if(FAllGrtr(pens[a], pen))
						{
							for(PxU32 b = a + 1; b < NB_TO_ADD; ++b)
							{
								pens[b] = pens[b - 1];
								inds[b] = inds[b - 1];
							}
							pens[a] = pen;
							inds[a] = i;
							break;
						}
					}
				}
			}
			currentPatch = currentPatch->mNextPatch;
		}
		for(PxU32 i = 0; i < NB_TO_ADD; ++i)
		{
			mContactPoints[i + 4] = manifoldContactExt[inds[i]];
			maxPen = FMin(maxPen, pens[i]);
		}
	}
	return maxPen;
}

EHDI PxU32 SinglePersistentContactManifold::reduceContacts(MeshPersistentContact* manifoldPoints, PxU32 numPoints)
{
	PxU8 chosenIndices[GU_SINGLE_MANIFOLD_SINGLE_POLYGONE_CACHE_SIZE];
	PxU8 candidates_buf[E_ALLOCA_N]; PxU8* candidates = candidates_buf;

	//(1) Remove duplicates...
	for(PxU32 i = 0; i < numPoints; ++i)
	{
		Vec3V localPointB = manifoldPoints[i].mLocalPointB;
		for(PxU32 j = i+1; j < numPoints; ++j)
		{
			if(V3AllGrtr(V3Eps(), V3Abs(V3Sub(localPointB, manifoldPoints[j].mLocalPointB))))
			{
				manifoldPoints[j] = manifoldPoints[numPoints - 1];
				j--;
				numPoints--;
			}
		}
	}

	if(numPoints <= GU_SINGLE_MANIFOLD_SINGLE_POLYGONE_CACHE_SIZE)
		return numPoints;

	const FloatV max = FMax();
	const FloatV nmax = FNeg(max);
	FloatV maxPen = V4GetW(manifoldPoints[0].mLocalNormalPen);
	PxU32 index = 0;
	candidates[0] = 0;
	PxU32 candidateIndex = 0;
	PxU32 nbCandiates = numPoints;

	MeshPersistentContact newManifold[GU_SINGLE_MANIFOLD_SINGLE_POLYGONE_CACHE_SIZE];

	//keep the deepest point
	for(PxU32 i = 1; i<numPoints; ++i)
	{
		//at the begining candidates and indices will be the same
		candidates[i] = PxU8(i);

		const FloatV pen = V4GetW(manifoldPoints[i].mLocalNormalPen);
		if(FAllGrtr(maxPen, pen))
		{
			maxPen = pen;
			index = i;
			candidateIndex = i;
		}
	}

	//keep the deepest points in the first position
	chosenIndices[0] = PxU8(index);

	//move the chosen indices out of the candidates indices
	nbCandiates = nbCandiates - 1;
	candidates[candidateIndex] = candidates[nbCandiates];

	//keep the deepest points in the first position
	newManifold[0] = manifoldPoints[chosenIndices[0]];

	//calculate the furthest away points
	Vec3V v = V3Sub(manifoldPoints[candidates[0]].mLocalPointB, newManifold[0].mLocalPointB);
	maxPen = V3Dot(v, v);
	index = candidates[0];
	candidateIndex = 0;

	for(PxU32 i = 1; i<nbCandiates; ++i)
	{
		v = V3Sub(manifoldPoints[candidates[i]].mLocalPointB, manifoldPoints[chosenIndices[0]].mLocalPointB);
		const FloatV d = V3Dot(v, v);
		if(FAllGrtr(d, maxPen))
		{
			maxPen = d;
			index = candidates[i];
			candidateIndex = i;
		}
	}

	chosenIndices[1] = PxU8(index);
	//move the chosen indices out of the candidates indices
	nbCandiates = nbCandiates - 1;
	candidates[candidateIndex] = candidates[nbCandiates];

	newManifold[1] = manifoldPoints[chosenIndices[1]];

	v = V3Sub(newManifold[1].mLocalPointB, newManifold[0].mLocalPointB);
	const Vec3V cn0 = Vec3V_From_Vec4V(newManifold[0].mLocalNormalPen);
	Vec3V norm = V3Cross(v, cn0);

	FloatV maxDist = nmax;
	FloatV minDist = max;
	index = GU_MANIFOLD_INVALID_INDEX;
	PxU32 index1 = GU_MANIFOLD_INVALID_INDEX;
	PxU32 candidateIndex1 = GU_MANIFOLD_INVALID_INDEX;

	//calculate the point furthest way to the segment
	for(PxU32 i = 0; i<nbCandiates; ++i)
	{
		v = V3Sub(manifoldPoints[candidates[i]].mLocalPointB, newManifold[0].mLocalPointB);
		const FloatV d = V3Dot(v, norm);
		if(FAllGrtr(d, maxDist))
		{
			maxDist = d;
			index = candidates[i];
			candidateIndex = i;
		}

		if(FAllGrtr(minDist, d))
		{
			minDist = d;
			index1 = candidates[i];
			candidateIndex1 = i;
		}
	}

	chosenIndices[2] = PxU8(index);
	//move the chosen indices out of the candidates indices
	nbCandiates = nbCandiates - 1;
	candidates[candidateIndex] = candidates[nbCandiates];

	newManifold[2] = manifoldPoints[chosenIndices[2]];

	if(nbCandiates == candidateIndex1)
		candidateIndex1 = candidateIndex;

	const FloatV temp = FMul(minDist, maxDist);
	if(FAllGrtr(temp, FZero()))
	{
		//chose again
		maxDist = nmax;
		for(PxU32 i = 0; i < nbCandiates; ++i)
		{
			v = V3Sub(manifoldPoints[candidates[i]].mLocalPointB, newManifold[0].mLocalPointB);
			const FloatV d = V3Dot(v, norm);
			if(FAllGrtr(d, maxDist))
			{
				maxDist = d;
				index1 = candidates[i];
				candidateIndex1 = i;
			}
		}
	}

	chosenIndices[3] = PxU8(index1);
	//move the chosen indices out of the candidates indices
	nbCandiates = nbCandiates - 1;
	candidates[candidateIndex1] = candidates[nbCandiates];

	newManifold[3] = manifoldPoints[chosenIndices[3]];

	maxDist = max;
	index = GU_MANIFOLD_INVALID_INDEX;
	candidateIndex = GU_MANIFOLD_INVALID_INDEX;
	//choose the 5 point, second deepest in the left overlap point
	for(PxU32 i = 0; i < nbCandiates; ++i)
	{
		const FloatV pen = V4GetW(manifoldPoints[candidates[i]].mLocalNormalPen);
		if(FAllGrtr(maxDist, pen))
		{
			maxDist = pen;
			index = candidates[i];
			candidateIndex = i;
		}
	}

	chosenIndices[4] = PxU8(index);
	//move the chosen indices out of the candidates indices
	nbCandiates = nbCandiates - 1;
	candidates[candidateIndex] = candidates[nbCandiates];

	newManifold[4] = manifoldPoints[chosenIndices[4]];

	//copy the new manifold back
	for(PxU32 i = 0; i<GU_SINGLE_MANIFOLD_SINGLE_POLYGONE_CACHE_SIZE; ++i)
	{
		manifoldPoints[i] = newManifold[i];
	}

	return GU_SINGLE_MANIFOLD_SINGLE_POLYGONE_CACHE_SIZE;
}

EHDI FloatV SinglePersistentContactManifold::refreshContactPoints(const PxMatTransformV& aToB, const FloatVArg projectBreakingThreshold, const FloatVArg /*contactOffset*/)
{
	const FloatV sqProjectBreakingThreshold = FMul(projectBreakingThreshold, projectBreakingThreshold); 

	FloatV maxPen = FZero();
	// first refresh worldspace positions and distance
	for(PxU32 i=mNumContacts; i > 0; --i)
	{
		MeshPersistentContact& manifoldPoint = mContactPoints[i-1];
		const Vec3V localAInB = aToB.transform( manifoldPoint.mLocalPointA ); // from a to b
		const Vec3V localBInB = manifoldPoint.mLocalPointB;
		const Vec3V v = V3Sub(localAInB, localBInB); 

		const Vec3V localNormal = Vec3V_From_Vec4V(manifoldPoint.mLocalNormalPen); // normal in b space
		const FloatV dist = V3Dot(v, localNormal);

		const Vec3V projectedPoint = V3NegScaleSub(localNormal, dist, localAInB);//manifoldPoint.worldPointA - manifoldPoint.worldPointB * manifoldPoint.m_distance1;
		const Vec3V projectedDifference = V3Sub(localBInB, projectedPoint);

		const FloatV distance2d = V3Dot(projectedDifference, projectedDifference);
		//const BoolV con = BOr(FIsGrtr(dist, contactOffset), FIsGrtr(distance2d, sqProjectBreakingThreshold));
		const BoolV con = FIsGrtr(distance2d, sqProjectBreakingThreshold);
		if(BAllEqTTTT(con))
		{
			removeContactPoint(i-1);
		} 
		else
		{
			manifoldPoint.mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), dist);
			maxPen = FMin(maxPen, dist);
		}
	}

	return maxPen;
}

EHDI void SinglePersistentContactManifold::drawManifold(PxRenderOutput& out, const PxTransformV& trA, const PxTransformV& trB) const
{
	PxVec3 a, b;
	V3StoreU(trA.p, a);
	V3StoreU(trB.p, b);

	for(PxU32 i = 0; i< mNumContacts; ++i)
	{
		const MeshPersistentContact& m = mContactPoints[i];
		drawManifoldPoint(m, trA, trB, out, gColors[i]);
	}
}

EHDI void MultiplePersistentContactManifold::drawManifold(PxRenderOutput& out, const PxTransformV& trA, const PxTransformV& trB) const
{
	for(PxU32 i=0; i<mNumManifolds; ++i)
	{
		const SinglePersistentContactManifold* manifold = getManifold(i);
		manifold->drawManifold(out, trA, trB);
	}
}

EHDI void MultiplePersistentContactManifold::drawLine(PxRenderOutput& out, const Vec3VArg p0, const Vec3VArg p1, PxU32 color)
{
	PxVec3 a, b;
	V3StoreU(p0, a);
	V3StoreU(p1, b);

	const PxMat44 m(PxIdentity);
	out << color << m << PxRenderOutput::LINES << a << b;
}

EHDI void MultiplePersistentContactManifold::drawLine(PxRenderOutput& out, const PxVec3 p0, const PxVec3 p1, PxU32 color)
{
	const PxMat44 m(PxIdentity);
	out << color << m << PxRenderOutput::LINES << p0 << p1;
}

EHDI void MultiplePersistentContactManifold::drawPoint(PxRenderOutput& out, const Vec3VArg p, const PxF32 size, PxU32 color)
{
	const PxVec3 up(0.f, size, 0.f);
	const PxVec3 right(size, 0.f, 0.f);
	const PxVec3 forwards(0.f, 0.f, size);

	PxVec3 a;
	V3StoreU(p, a);

	const PxMat44 m(PxIdentity);
	
	out << color << m << PxRenderOutput::LINES << a + up << a - up;
	out << color << m << PxRenderOutput::LINES << a + right << a - right;
	out << color << m << PxRenderOutput::LINES << a + forwards << a - forwards;
}

EHDI void MultiplePersistentContactManifold::drawPolygon(PxRenderOutput& out, const PxTransformV& transform, Vec3V* points, PxU32 numVerts, PxU32 color)
{
	for(PxU32 i=0; i<numVerts; ++i)
	{
		Vec3V tempV0 = points[i == 0 ? numVerts-1 : i-1];
		Vec3V tempV1 = points[i];
		
		drawLine(out, transform.transform(tempV0), transform.transform(tempV1), color);
	}
}

EHDI static FloatV addBatchManifoldContactsToSingleManifold(SinglePersistentContactManifold* manifold, MeshPersistentContact* manifoldContact, PxU32 numManifoldContacts, PCMContactPatch* patch, const FloatVArg sqReplaceBreakingThreshold, PxU8 maxContactsPerManifold)
{
	switch(maxContactsPerManifold)
	{
	case GU_SPHERE_MANIFOLD_CACHE_SIZE://sphere
		return manifold->addBatchManifoldContactsSphere(manifoldContact, numManifoldContacts, *patch, sqReplaceBreakingThreshold);
	case GU_CAPSULE_MANIFOLD_CACHE_SIZE://capsule, need to implement keep two deepest
		return manifold->addBatchManifoldContactsCapsule(manifoldContact, numManifoldContacts, *patch, sqReplaceBreakingThreshold);
	default://cache size GU_SINGLE_MANIFOLD_CACHE_SIZE
		return manifold->addBatchManifoldContactsConvex(manifoldContact, numManifoldContacts, *patch, sqReplaceBreakingThreshold);
	};
}

//	This function adds the manifold contacts with different patches into the corresponding single persistent contact manifold
EHDI void MultiplePersistentContactManifold::addManifoldContactPoints(MeshPersistentContact* manifoldContact, PxU32 numManifoldContacts, PCMContactPatch** contactPatch, PxU32 numPatch, const FloatVArg sqReplaceBreakingThreshold, const FloatVArg acceptanceEpsilon, PxU8 maxContactsPerManifold)
{
	if(mNumManifolds == 0)
	{
		for(PxU32 i=0; i<numPatch; ++i)
		{
			PCMContactPatch* patch = contactPatch[i];
			//this mean the patch hasn't been add to the manifold
			if(patch->mRoot == patch)
			{
				
				SinglePersistentContactManifold* manifold = getEmptyManifold();
				if(manifold)
				{
					const FloatV _maxPen = addBatchManifoldContactsToSingleManifold(manifold, manifoldContact, numManifoldContacts, patch, sqReplaceBreakingThreshold, maxContactsPerManifold);
					FStore(_maxPen, &mMaxPen[mManifoldIndices[mNumManifolds]]);
					mNumManifolds++;
				}
				else
				{
					//We already pre-sorted the patches so we know we can return here
					return;
				}
			}
		}
	}
	else
	{
		//we do processContacts() when the number of contacts are more than 16, such that, we might call processContacts() multiple times when we have very detailed mesh
		//or very large contact offset/objects. In this case, the mNumManifolds will be large than 0. 
		PCMContactPatch tempPatch;
		for(PxU32 i=0; i<numPatch; ++i)
		{
			bool found = false;
			PCMContactPatch* patch = contactPatch[i];
			//this mean the patch has't been add to the manifold
			if(patch->mRoot == patch)
			{
				E_ASSERT(mNumManifolds <= GU_MAX_MANIFOLD_SIZE);
				for(PxU32 j=0; j<mNumManifolds; ++j)
				{
					E_ASSERT(mManifoldIndices[j] < GU_MAX_MANIFOLD_SIZE);
					SinglePersistentContactManifold& manifold = *getManifold(j);

					const Vec3V pNor = manifold.getLocalNormal();
					const FloatV d = V3Dot(patch->mPatchNormal, pNor);

					if(FAllGrtrOrEq(d, acceptanceEpsilon))
					{
						//appending the existing contacts to the manifold contact stream
						for(PxU32 k=0; k< manifold.mNumContacts; ++k)
						{
							PxU32 index = k + numManifoldContacts;
							//not duplicate point
							E_ASSERT(index < 64);
							manifoldContact[index] = manifold.mContactPoints[k];
						}

						//create a new patch for the exiting manifold
						tempPatch.mStartIndex = numManifoldContacts;
						tempPatch.mEndIndex = numManifoldContacts + manifold.mNumContacts;
						tempPatch.mPatchNormal = pNor;//manifold.getLocalNormal();
						tempPatch.mRoot = patch;
						tempPatch.mNextPatch = NULL;
						
						patch->mEndPatch->mNextPatch = &tempPatch;
				
						//numManifoldContacts += manifold.numContacts;
						patch->mTotalSize += manifold.mNumContacts;
						patch->mPatchMaxPen = FMin(patch->mPatchMaxPen, FLoad(mMaxPen[mManifoldIndices[j]]));
					
						//manifold.numContacts = 0;

						E_ASSERT((numManifoldContacts+manifold.mNumContacts) <= 64);
						const FloatV _maxPen = addBatchManifoldContactsToSingleManifold(&manifold, manifoldContact, numManifoldContacts+manifold.mNumContacts, patch, sqReplaceBreakingThreshold, maxContactsPerManifold);
						FStore(_maxPen, &mMaxPen[mManifoldIndices[j]]);
						found = true;
						break;
					}
				}

				if(!found)// && numManifolds < 4)
				{
					SinglePersistentContactManifold* manifold = getEmptyManifold();
					//we still have slot to create a new manifold
					if(manifold)
					{
						const FloatV _maxPen = addBatchManifoldContactsToSingleManifold(manifold, manifoldContact, numManifoldContacts, patch, sqReplaceBreakingThreshold, maxContactsPerManifold);
						FStore(_maxPen, &mMaxPen[mManifoldIndices[mNumManifolds]]);
						mNumManifolds++;
					}	
					else
					{
						//we can't allocate a new manifold  and no existing manifold has the same normal as this patch, we need to find the shallowest penetration manifold. If this manifold is shallower than
						//the current patch, replace the manifold with the current patch
						PxU32 index = 0;
						for(PxU32 j=1; j<mNumManifolds; ++j)
						{
							//if(FAllGrtr(mMaxPen[mManifoldIndices[i]], mMaxPen[mManifoldIndices[index]]))
							if(mMaxPen[mManifoldIndices[j]] > mMaxPen[mManifoldIndices[index]])
							{
								index = j;
							}
						}

						if(FAllGrtr(FLoad(mMaxPen[mManifoldIndices[index]]), patch->mPatchMaxPen))
						{
							manifold = getManifold(index);
							manifold->mNumContacts = 0;
							const FloatV _maxPen = addBatchManifoldContactsToSingleManifold(manifold, manifoldContact, numManifoldContacts, patch, sqReplaceBreakingThreshold, maxContactsPerManifold);
							FStore(_maxPen, &mMaxPen[mManifoldIndices[index]]);

						}
						return;
					}
				}
			}
		}
	}
}

//This function adds the multi manifold contacts to the contact buffer for box/convexhull
EHDI bool MultiplePersistentContactManifold::addManifoldContactsToContactBuffer(PxContactBuffer& contactBuffer, const PxTransformV& meshTransform)
{
	PxU32 contactCount = 0;
	PxU32 numContacts = 0;
	mNumTotalContacts = 0;
	//drawManifold(*gRenderOutPut, convexTransform, meshTransform);
	for(PxU32 i=0; i < mNumManifolds; ++i)
	{
		SinglePersistentContactManifold& manifold = *getManifold(i);
		numContacts = manifold.getNumContacts();
		E_ASSERT(mNumTotalContacts + numContacts <= 0xFF);
		mNumTotalContacts += PxTo8(numContacts);
		const Vec3V normal = manifold.getWorldNormal(meshTransform);
		
		for(PxU32 j=0; (j< numContacts) & (contactCount < PxContactBuffer::MAX_CONTACTS); ++j)
		{
			const MeshPersistentContact& p = manifold.getContactPoint(j);
			
			const Vec3V worldP = meshTransform.transform(p.mLocalPointB);
			const FloatV dist = V4GetW(p.mLocalNormalPen);
			
			outputPCMContact(contactBuffer, contactCount, worldP, normal, dist, p.mFaceIndex);
		}
	}

	E_ASSERT(contactCount <= PxContactBuffer::MAX_CONTACTS);
	contactBuffer.count = contactCount;
	return contactCount > 0;
}

//This function adds the multi manifold contacts to the contact buffer for sphere and capsule, radius is corresponding to the sphere radius/capsule radius
EHDI bool MultiplePersistentContactManifold::addManifoldContactsToContactBuffer(PxContactBuffer& contactBuffer, const PxTransformV& trA, const PxTransformV& trB, const FloatVArg radius)
{
	PxU32 contactCount = 0;
	PxU32 numContacts = 0;
	mNumTotalContacts = 0;

	for(PxU32 i=0; i < mNumManifolds; ++i)
	{
		//get a single manifold
		SinglePersistentContactManifold& manifold = *getManifold(i);
		numContacts = manifold.getNumContacts();
		E_ASSERT(mNumTotalContacts + numContacts <= 0xFF);
		mNumTotalContacts += PxTo8(numContacts);
		const Vec3V normal = manifold.getWorldNormal(trB);
		
		//iterate all the contacts in this single manifold and add contacts to the contact buffer
		//each manifold contact point store two points which are in each other's local space. In this
		//case, mLocalPointA is stored as the center of sphere/a point in the segment for capsule
		for(PxU32 j=0; (j< numContacts) & (contactCount < PxContactBuffer::MAX_CONTACTS); ++j)
		{
			const MeshPersistentContact& p = manifold.getContactPoint(j);
			
			const Vec3V worldP = V3NegScaleSub(normal, radius, trA.transform(p.mLocalPointA));
			const FloatV dist = FSub(V4GetW(p.mLocalNormalPen), radius);
			
			outputPCMContact(contactBuffer, contactCount, worldP, normal, dist, p.mFaceIndex);
		}
	}

	E_ASSERT(contactCount <= PxContactBuffer::MAX_CONTACTS);
	contactBuffer.count = contactCount;
	return contactCount > 0;
}

//	This function copies the mesh persistent contact from compress buffer(NpCacheStreamPair in the PxcNpThreadContext) to the multiple manifold
// PT: function moved to cpp to go around a compiler bug on PS4
EHDI void MultiplePersistentContactManifold::fromBuffer(PxU8*  buffer)
{
	PxU32 numManifolds = 0;
	if (buffer != NULL)
	{
		E_ASSERT(((uintptr_t(buffer)) & 0xF) == 0);
		PxU8*  buff = buffer;
		const MultiPersistentManifoldHeader*  header = reinterpret_cast<const MultiPersistentManifoldHeader*>(buff);
		buff += sizeof(MultiPersistentManifoldHeader);

		numManifolds = header->mNumManifolds;

		E_ASSERT(numManifolds <= GU_MAX_MANIFOLD_SIZE);
		mRelativeTransform = header->mRelativeTransform;

		for (PxU32 a = 0; a < numManifolds; ++a)
		{
			mManifoldIndices[a] = PxU8(a);
			const SingleManifoldHeader*  manHeader = reinterpret_cast<const SingleManifoldHeader*>(buff);
			buff += sizeof(SingleManifoldHeader);
			const PxU32 numContacts = manHeader->mNumContacts;
			E_ASSERT(numContacts <= GU_SINGLE_MANIFOLD_CACHE_SIZE);
			SinglePersistentContactManifold& manifold = mManifolds[a];
			manifold.mNumContacts = numContacts;
			E_ASSERT((uintptr_t(buff) & 0xf) == 0);
			const CachedMeshPersistentContact* contacts = reinterpret_cast<const CachedMeshPersistentContact*>(buff);
			for (PxU32 b = 0; b < manifold.mNumContacts; ++b)
			{
				manifold.mContactPoints[b].mLocalPointA = Vec3V_From_Vec4V(V4LoadA(&contacts[b].mLocalPointA.x));
				manifold.mContactPoints[b].mLocalPointB = Vec3V_From_Vec4V(V4LoadA(&contacts[b].mLocalPointB.x));
				manifold.mContactPoints[b].mLocalNormalPen = V4LoadA(&contacts[b].mLocalNormal.x);
				manifold.mContactPoints[b].mFaceIndex = contacts[b].mFaceIndex;
			}
			buff += sizeof(CachedMeshPersistentContact) * numContacts;
		}
	}
	else
	{
		mRelativeTransform.invalidate();
	}
	mNumManifolds = PxU8(numManifolds);
	for (PxU32 a = numManifolds; a < GU_MAX_MANIFOLD_SIZE; ++a)
	{
		mManifoldIndices[a] = PxU8(a);
	}
}

// This function copies the mesh persistent contact from the multiple manifold to compress buffer(NpCacheStreamPair in the PxcNpThreadContext)
EHDI void MultiplePersistentContactManifold::toBuffer(PxU8*  buffer) const
{
	PxU8* buff = buffer;

	E_ASSERT(((uintptr_t(buff)) & 0xF) == 0);
	MultiPersistentManifoldHeader*  header = reinterpret_cast<MultiPersistentManifoldHeader*>(buff);
	buff += sizeof(MultiPersistentManifoldHeader);

	E_ASSERT(mNumManifolds <= GU_MAX_MANIFOLD_SIZE);
	header->mNumManifolds = mNumManifolds;
	header->mRelativeTransform = mRelativeTransform;

	for(PxU32 a = 0; a < mNumManifolds; ++a)
	{
		SingleManifoldHeader* manHeader = reinterpret_cast<SingleManifoldHeader*>(buff);
		buff += sizeof(SingleManifoldHeader);
		const SinglePersistentContactManifold& manifold = *getManifold(a);
		manHeader->mNumContacts = manifold.mNumContacts;
		E_ASSERT((uintptr_t(buff) & 0xf) == 0);
		CachedMeshPersistentContact* contacts = reinterpret_cast<CachedMeshPersistentContact*>(buff);
		//convert the mesh persistent contact to cached mesh persistent contact to save 16 byte memory per contact
		for(PxU32 b = 0; b<manifold.mNumContacts; ++b)
		{	
			V4StoreA(Vec4V_From_Vec3V(manifold.mContactPoints[b].mLocalPointA), &contacts[b].mLocalPointA.x);
			V4StoreA(Vec4V_From_Vec3V(manifold.mContactPoints[b].mLocalPointB), &contacts[b].mLocalPointB.x);
			V4StoreA(manifold.mContactPoints[b].mLocalNormalPen, &contacts[b].mLocalNormal.x);
			//Note - this must be written last because we just wrote mLocalPointA to this memory so need to make sure
			//that face index is written after that.
			contacts[b].mFaceIndex = manifold.mContactPoints[b].mFaceIndex;
		}
		buff += sizeof(CachedMeshPersistentContact) * manifold.mNumContacts;
	}
}

EHDI void Gu::addManifoldPoint(PersistentContact* manifoldContacts, PersistentContactManifold& manifold, GjkOutput& output,
	const PxMatTransformV& aToB, const FloatV replaceBreakingThreshold)
{
	const Vec3V localPointA = aToB.transformInv(output.closestA);
	const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(output.normal), output.penDep);

	//Add contact to contact stream
	manifoldContacts[0].mLocalPointA = localPointA;
	manifoldContacts[0].mLocalPointB = output.closestB;
	manifoldContacts[0].mLocalNormalPen = localNormalPen;

	//Add contact to manifold
	manifold.addManifoldPoint(localPointA, output.closestB, localNormalPen, replaceBreakingThreshold);
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMShapeConvex.h
namespace px
{
	class PxConvexMeshGeometry;

namespace Gu
{
	E_GTABLE_DECL(PxU8, gPCMBoxPolygonData);
	
	class PCMPolygonalBox
	{
	public:
									EHDV PCMPolygonalBox(const PxVec3& halfSide);

			EHDV void					getPolygonalData(Gu::PolygonalData*  dst)	const;

			const PxVec3&			mHalfSide;
			PxVec3					mVertices[8];
			Gu::HullPolygonData		mPolygons[6];
	private:
			PCMPolygonalBox& operator=(const PCMPolygonalBox&);
	};

	EHDV void getPCMConvexData(const Gu::ConvexHullV& convexHull, bool idtScale, Gu::PolygonalData& polyData);
	EHDV bool getPCMConvexData(const PxConvexMeshGeometry& shapeConvex, Cm::FastVertex2ShapeScaling& scaling, PxBounds3& bounds, Gu::PolygonalData& polyData);
}
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMShapeConvex.cpp
using namespace px;
using namespace Gu;
using namespace aos;

namespace px
{
namespace Gu
{
E_GTABLE(PxU8, gPCMBoxPolygonData, 24, { 0, 3, 2, 1, 1, 2, 6, 5, 5, 6, 7, 4, 4, 7, 3, 0, 3, 7, 6, 2, 4, 0, 1, 5, })
}
}

EHDI PCMPolygonalBox::PCMPolygonalBox(const PxVec3& halfSide) : mHalfSide(halfSide)
{
	//Precompute the convex data
	//     7+------+6			0 = ---
	//     /|     /|			1 = +--
	//    / |    / |			2 = ++-
	//   / 4+---/--+5			3 = -+-
	// 3+------+2 /    y   z	4 = --+
	//  | /    | /     |  /		5 = +-+
	//  |/     |/      |/		6 = +++
	// 0+------+1      *---x	7 = -++

	PxVec3 minimum = -mHalfSide;
	PxVec3 maximum = mHalfSide;
	// Generate 8 corners of the bbox
	mVertices[0] = PxVec3(minimum.x, minimum.y, minimum.z);
	mVertices[1] = PxVec3(maximum.x, minimum.y, minimum.z);
	mVertices[2] = PxVec3(maximum.x, maximum.y, minimum.z);
	mVertices[3] = PxVec3(minimum.x, maximum.y, minimum.z);
	mVertices[4] = PxVec3(minimum.x, minimum.y, maximum.z);
	mVertices[5] = PxVec3(maximum.x, minimum.y, maximum.z);
	mVertices[6] = PxVec3(maximum.x, maximum.y, maximum.z);
	mVertices[7] = PxVec3(minimum.x, maximum.y, maximum.z);

	//Setup the polygons
	for(PxU8 i=0; i < 6; i++)
	{
		mPolygons[i].mNbVerts = 4;
		mPolygons[i].mVRef8 = PxU16(i*4);
	}

	// ### planes needs *very* careful checks
	// X axis
	mPolygons[1].mPlane.n = PxVec3(1.0f, 0.0f, 0.0f);
	mPolygons[1].mPlane.d = -mHalfSide.x;
	mPolygons[3].mPlane.n = PxVec3(-1.0f, 0.0f, 0.0f);
	mPolygons[3].mPlane.d = -mHalfSide.x;
		
	mPolygons[1].mMinIndex = 0;
	mPolygons[3].mMinIndex = 1;

	E_ASSERT(mPolygons[1].getMin(mVertices) == -mHalfSide.x); 
	E_ASSERT(mPolygons[3].getMin(mVertices) == -mHalfSide.x);

	// Y axis
	mPolygons[4].mPlane.n = PxVec3(0.f, 1.0f, 0.0f);
	mPolygons[4].mPlane.d = -mHalfSide.y;
	mPolygons[5].mPlane.n = PxVec3(0.0f, -1.0f, 0.0f);
	mPolygons[5].mPlane.d = -mHalfSide.y;

	mPolygons[4].mMinIndex = 0;
	mPolygons[5].mMinIndex = 2;

	E_ASSERT(mPolygons[4].getMin(mVertices) == -mHalfSide.y); 
	E_ASSERT(mPolygons[5].getMin(mVertices) == -mHalfSide.y);

	// Z axis
	mPolygons[2].mPlane.n = PxVec3(0.f, 0.0f, 1.0f);
	mPolygons[2].mPlane.d = -mHalfSide.z;
	mPolygons[0].mPlane.n = PxVec3(0.0f, 0.0f, -1.0f);
	mPolygons[0].mPlane.d = -mHalfSide.z;

	mPolygons[2].mMinIndex = 0;
	mPolygons[0].mMinIndex = 4;
	E_ASSERT(mPolygons[2].getMin(mVertices) == -mHalfSide.z); 
	E_ASSERT(mPolygons[0].getMin(mVertices) == -mHalfSide.z);
}

EHDI void PCMPolygonalBox::getPolygonalData(PolygonalData*  dst) const
{
	dst->mCenter				= PxVec3(0.0f, 0.0f, 0.0f);
	dst->mNbVerts				= 8;
	dst->mNbPolygons			= 6;
	dst->mPolygons				= mPolygons;
	dst->mNbEdges				= 0;
	dst->mVerts					= mVertices;
	dst->mPolygonVertexRefs		= gPCMBoxPolygonData_p();
	dst->mFacesByEdges			= NULL;
	dst->mVerticesByEdges		= NULL;

	dst->mBigData				= NULL;

	dst->mInternal.mInternalRadius	= 0.0f;
	dst->mInternal.mInternalExtents	= mHalfSide;

	dst->mScale = PxMeshScale();
}

EHDI static void getPCMPolygonalData_Convex(PolygonalData*  dst, const ConvexHullData*  src, const Mat33V& vertexToShape)
{
	using namespace aos;
	const Vec3V vertexSpaceCenterOfMass = V3LoadU(src->mCenterOfMass);
	const Vec3V shapeSpaceCenterOfMass = M33MulV3(vertexToShape, vertexSpaceCenterOfMass);
	V3StoreU(shapeSpaceCenterOfMass, dst->mCenter);
	dst->mNbVerts			= src->mNbHullVertices;
	dst->mNbPolygons		= src->mNbPolygons;
	dst->mNbEdges			= src->mNbEdges;
	dst->mPolygons			= src->mPolygons;
	dst->mVerts				= src->getHullVertices();
	dst->mPolygonVertexRefs	= src->getVertexData8();
	dst->mFacesByEdges		= src->getFacesByEdges8();
	dst->mVerticesByEdges   = src->getVerticesByEdges16();

	dst->mBigData			= src->mBigConvexRawData;

	dst->mInternal			= src->mInternal;
		
	dst->mScale				= PxMeshScale();
}

EHDI static void getPCMPolygonalData_Convex(PolygonalData*  dst, const ConvexHullData*  src, 
	const Cm::FastVertex2ShapeScaling& scaling, const PxMeshScale& convexScale)
{
	dst->mCenter			= scaling * src->mCenterOfMass;
	dst->mNbVerts			= src->mNbHullVertices;
	dst->mNbPolygons		= src->mNbPolygons;
	dst->mNbEdges			= src->mNbEdges;
	dst->mPolygons			= src->mPolygons;
	dst->mVerts				= src->getHullVertices();
	dst->mPolygonVertexRefs	= src->getVertexData8();
	dst->mFacesByEdges		= src->getFacesByEdges8();
	dst->mVerticesByEdges   = src->getVerticesByEdges16();

	dst->mBigData			= src->mBigConvexRawData;
	dst->mInternal			= src->mInternal;
	dst->mScale				= convexScale;
}

EHDI void Gu::getPCMConvexData(const ConvexHullV& convexHull, bool idtScale, PolygonalData& polyData)
{
	E_ASSERT(!convexHull.hullData->mAABB.isEmpty());  

	//this is used to calculate the convex hull's center of mass
	getPCMPolygonalData_Convex(&polyData, convexHull.hullData, convexHull.vertex2Shape);

	if(!idtScale)
		polyData.mInternal.reset();
}
     
EHDI bool Gu::getPCMConvexData(const PxConvexMeshGeometry& shapeConvex, Cm::FastVertex2ShapeScaling& scaling, PxBounds3& bounds, PolygonalData& polyData)
{
	const bool idtScale = shapeConvex.scale.isIdentity();
	if(!idtScale)
		scaling.init(shapeConvex.scale);

	const ConvexHullData* hullData = _getHullData(shapeConvex);
	E_ASSERT(!hullData->mAABB.isEmpty());
	bounds = hullData->mAABB.transformFast(scaling.getVertex2ShapeSkew());

	getPCMPolygonalData_Convex(&polyData, hullData, scaling, shapeConvex.scale);

	return idtScale;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactGenUtil.cpp
using namespace px;
using namespace Gu;
using namespace aos;

EHDI bool Gu::contains(Vec3V* verts, PxU32 numVerts, const Vec3VArg p, const Vec3VArg min, const Vec3VArg max)
{
	const BoolV tempCon = BOr(V3IsGrtr(min, p), V3IsGrtr(p, max));
	const BoolV con = BOr(BGetX(tempCon), BGetY(tempCon));

	if (BAllEqTTTT(con))
		return false;

	const FloatV tx = V3GetX(p);
	const FloatV ty = V3GetY(p);

	const FloatV eps = FEps();
	const FloatV zero = FZero();
	PxU32 intersectionPoints = 0;

	PxU32 i = 0, j = numVerts - 1;

	for (; i < numVerts; j = i++)
	{
		const FloatV jy = V3GetY(verts[j]);
		const FloatV iy = V3GetY(verts[i]);

		const FloatV jx = V3GetX(verts[j]);
		const FloatV ix = V3GetX(verts[i]);

		//if p is one of the end point, we will return intersect
		const BoolV con0 = BAnd(FIsEq(tx, jx), FIsEq(ty, jy));
		const BoolV con1 = BAnd(FIsEq(tx, ix), FIsEq(ty, iy));

		if (BAllEqTTTT(BOr(con0, con1)))
			return true;

		//(verts[i].y > test.y) != (points[j].y > test.y) 
		const PxU32 yflag0 = FAllGrtr(jy, ty);
		const PxU32 yflag1 = FAllGrtr(iy, ty);

		//ML: the only case the ray will intersect this segment is when the p's y is in between two segments y
		if (yflag0 != yflag1)
		{
			//ML: choose ray, which start at p and every points in the ray will have the same y component
			//t1 = (yp - yj)/(yi - yj)
			//qx = xj + t1*(xi-xj)
			//t = qx - xp > 0 for the ray and segment intersection happen
			const FloatV jix = FSub(ix, jx);
			const FloatV jiy = FSub(iy, jy);
			//const FloatV jtx = FSub(tx, jy);
			const FloatV jty = FSub(ty, jy);
			const FloatV part1 = FMul(jty, jix);
			//const FloatV part2 = FMul(jx, jiy);
			//const FloatV part3 = FMul(V3Sub(tx, eps), jiy);
			const FloatV part2 = FMul(FAdd(jx, eps), jiy);
			const FloatV part3 = FMul(tx, jiy);

			const BoolV comp = FIsGrtr(jiy, zero);
			const FloatV tmp = FAdd(part1, part2);
			const FloatV comp1 = FSel(comp, tmp, part3);
			const FloatV comp2 = FSel(comp, part3, tmp);

			if (FAllGrtrOrEq(comp1, comp2))
			{
				if (intersectionPoints == 1)
					return false;

				intersectionPoints++;
			}
		}
	}
	return intersectionPoints> 0;
}

EHDI PxI32 Gu::getPolygonIndex(const PolygonalData& polyData, const SupportLocal* map, const Vec3VArg normal, PxI32& polyIndex2)
{
	//normal is in shape space, need to transform the vertex space
	const Vec3V n = M33TrnspsMulV3(map->vertex2Shape, normal);
	const Vec3V nnormal = V3Neg(n);
	const Vec3V planeN_ = V3LoadU_SafeReadW(polyData.mPolygons[0].mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class
	FloatV minProj = V3Dot(n, planeN_);

	const FloatV zero = FZero();
	PxI32 closestFaceIndex = 0;

	polyIndex2 = -1;

	for (PxU32 i = 1; i < polyData.mNbPolygons; ++i)
	{
		const Vec3V planeN = V3LoadU_SafeReadW(polyData.mPolygons[i].mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class
		const FloatV proj = V3Dot(n, planeN);
		if (FAllGrtr(minProj, proj))
		{
			minProj = proj;
			closestFaceIndex = PxI32(i);
		}
	}

	const PxU32 numEdges = polyData.mNbEdges;
	const PxU8* const edgeToFace = polyData.mFacesByEdges;

	//Loop through edges
	PxU32 closestEdge = 0xffffffff;
	FloatV maxDpSq = FMul(minProj, minProj);

	FloatV maxDpSq2 = FLoad(-10.0f);

	for (PxU32 i = 0; i < numEdges; ++i)//, inc = VecI32V_Add(inc, vOne))
	{
		const PxU32 index = i * 2;
		const PxU8 f0 = edgeToFace[index];
		const PxU8 f1 = edgeToFace[index + 1];

		const Vec3V planeNormal0 = V3LoadU_SafeReadW(polyData.mPolygons[f0].mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class
		const Vec3V planeNormal1 = V3LoadU_SafeReadW(polyData.mPolygons[f1].mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class

		// unnormalized edge normal
		const Vec3V edgeNormal = V3Add(planeNormal0, planeNormal1);//polys[f0].mPlane.n + polys[f1].mPlane.n;
		const FloatV enMagSq = V3Dot(edgeNormal, edgeNormal);//edgeNormal.magnitudeSquared();
		//Test normal of current edge - squared test is valid if dp and maxDp both >= 0
		const FloatV dp = V3Dot(edgeNormal, nnormal);//edgeNormal.dot(normal);
		const FloatV sqDp = FMul(dp, dp);

		if(f0==closestFaceIndex || f1==closestFaceIndex)
		{
			if(BAllEqTTTT(FIsGrtrOrEq(sqDp, maxDpSq2)))
			{
				maxDpSq2 = sqDp;
				polyIndex2 = f0==closestFaceIndex ? PxI32(f1) : PxI32(f0);
			}
		}

		const BoolV con0 = FIsGrtrOrEq(dp, zero);
		const BoolV con1 = FIsGrtr(sqDp, FMul(maxDpSq, enMagSq));
		const BoolV con = BAnd(con0, con1);
		if (BAllEqTTTT(con))
		{
			maxDpSq = FDiv(sqDp, enMagSq);
			closestEdge = i;
		}
	}

	if (closestEdge != 0xffffffff)
	{
		const PxU8* FBE = edgeToFace;

		const PxU32 index = closestEdge * 2;
		const PxU32 f0 = FBE[index];
		const PxU32 f1 = FBE[index + 1];

		const Vec3V planeNormal0 = V3LoadU_SafeReadW(polyData.mPolygons[f0].mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class
		const Vec3V planeNormal1 = V3LoadU_SafeReadW(polyData.mPolygons[f1].mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class

		const FloatV dp0 = V3Dot(planeNormal0, nnormal);
		const FloatV dp1 = V3Dot(planeNormal1, nnormal);
		if (FAllGrtr(dp0, dp1))
		{
			closestFaceIndex = PxI32(f0);
			polyIndex2 = PxI32(f1);
		}
		else
		{
			closestFaceIndex = PxI32(f1);
			polyIndex2 = PxI32(f0);
		}
	}

	return closestFaceIndex;
}

EHDI PxU32 Gu::getWitnessPolygonIndex(const PolygonalData& polyData, const SupportLocal* map, const Vec3VArg normal, const Vec3VArg closest, PxReal tolerance)
{
	PxReal pd[256];
	//first pass : calculate the smallest distance from the closest point to the polygon face

	//transform the closest p to vertex space
	const Vec3V p = M33MulV3(map->shape2Vertex, closest);
	PxU32 closestFaceIndex = 0;

	PxVec3 closestP;
	V3StoreU(p, closestP);

	const PxReal eps = -tolerance;

	PxPlane plane = polyData.mPolygons[0].mPlane;
	PxReal dist = plane.distance(closestP);
	PxReal minDist = dist >= eps ? PxAbs(dist) : EPX_MAX_F32;
	pd[0] = minDist;
	PxReal maxDist = dist;
	PxU32 maxFaceIndex = 0;

	for (PxU32 i = 1; i < polyData.mNbPolygons; ++i)
	{
		plane = polyData.mPolygons[i].mPlane;
		dist = plane.distance(closestP);
		pd[i] = dist >= eps ? PxAbs(dist) : EPX_MAX_F32;
		if (minDist > pd[i])
		{
			minDist = pd[i];
			closestFaceIndex = i;
		}
		if (dist > maxDist)
		{
			maxDist = dist;
			maxFaceIndex = i;
		}
	}

	if (minDist == EPX_MAX_F32)
		return maxFaceIndex;

	//second pass : select the face which has the normal most close to the gjk/epa normal
	Vec4V plane4 = V4LoadU(&polyData.mPolygons[closestFaceIndex].mPlane.n.x);
	Vec3V n = Vec3V_From_Vec4V(plane4);
	n = V3Normalize(M33TrnspsMulV3(map->shape2Vertex, n));
	FloatV bestProj = V3Dot(n, normal);

	const PxU32 firstPassIndex = closestFaceIndex;

	for (PxU32 i = 0; i< polyData.mNbPolygons; ++i)
	{
		//if the difference between the minimum distance and the distance of p to plane i is within tolerance, we use the normal to chose the best plane 
		if ((tolerance >(pd[i] - minDist)) && (firstPassIndex != i))
		{
			plane4 = V4LoadU(&polyData.mPolygons[i].mPlane.n.x);
			n = Vec3V_From_Vec4V(plane4);
			//rotate the plane normal to shape space. We can roate normal into the vertex space. The reason for
			//that is because it will lose some numerical precision if the object has large scale and this algorithm
			//will choose different face
			n = V3Normalize(M33TrnspsMulV3(map->shape2Vertex, n));
			FloatV proj = V3Dot(n, normal);

			if (FAllGrtr(bestProj, proj))
			{
				closestFaceIndex = i;
				bestProj = proj;
			}
		}
	}

	return closestFaceIndex;
}

/*	EHD bool boxContainsInXY(const FloatVArg x, const FloatVArg y, const Vec3VArg p, const Vec3V* verts, const Vec3VArg min, const Vec3VArg max)
	{
		using namespace aos;

		const BoolV tempCon = BOr(V3IsGrtr(min, p), V3IsGrtr(p, max));
		const BoolV con = BOr(BGetX(tempCon), BGetY(tempCon));
		
		if(BAllEqTTTT(con))
			return false;  
		
		const FloatV zero = FZero();
		FloatV PreviousX = V3GetX(verts[3]);
		FloatV PreviousY = V3GetY(verts[3]);

		// Loop through quad vertices
		for(PxI32 i=0; i<4; i++)
		{
			const FloatV CurrentX = V3GetX(verts[i]);
			const FloatV CurrentY = V3GetY(verts[i]);

			// |CurrentX - PreviousX      x - PreviousX|
			// |CurrentY - PreviousY      y - PreviousY|
			// => similar to backface culling, check each one of the 4 triangles are consistent, in which case
			// the point is within the parallelogram.
			const FloatV v00 = FSub(CurrentX, PreviousX);
			const FloatV v01 = FSub(y, PreviousY);
			const FloatV v10 = FSub(CurrentY, PreviousY);
			const FloatV v11 = FSub(x, PreviousX);
			const FloatV temp0 = FMul(v00, v01);
			const FloatV temp1 = FMul(v10, v11);
			if(FAllGrtrOrEq(FSub(temp0, temp1), zero))
				return false;

			PreviousX = CurrentX;
			PreviousY = CurrentY;
		}

		return true;
	}
*/

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactGen.h
namespace px
{
namespace Gu
{
	//full contact gen code for box/convexhull vs convexhull
	EHDV bool generateFullContactManifold(const Gu::PolygonalData& polyData0, const Gu::PolygonalData& polyData1, const Gu::SupportLocal* map0, const Gu::SupportLocal* map1, Gu::PersistentContact* manifoldContacts, 
		PxU32& numContacts, const aos::FloatVArg contactDist, const aos::Vec3VArg normal, const aos::Vec3VArg closestA, const aos::Vec3VArg closestB, 
		PxReal toleranceA, PxReal toleranceB, bool doOverlapTest, PxRenderOutput* renderOutput, PxReal toleranceLength);

	//full contact gen code for capsule vs convexhulll
	EHDV bool generateFullContactManifold(const Gu::CapsuleV& capsule, const Gu::PolygonalData& polyData, const Gu::SupportLocal* map, const aos::PxMatTransformV& aToB, Gu::PersistentContact* manifoldContacts, 
		PxU32& numContacts, const aos::FloatVArg contactDist, aos::Vec3V& normal, const aos::Vec3VArg closest, PxReal tolerance, bool doOverlapTest, PxReal toleranceScale);

	//full contact gen code for capsule vs box
	EHDV bool generateCapsuleBoxFullContactManifold(const Gu::CapsuleV& capsule, const Gu::PolygonalData& polyData, const Gu::SupportLocal* map, const aos::PxMatTransformV& aToB, Gu::PersistentContact* manifoldContacts, PxU32& numContacts,
		const aos::FloatVArg contactDist, aos::Vec3V& normal, const aos::Vec3VArg closest, PxReal boxMargin, bool doOverlapTest, PxReal toleranceScale, PxRenderOutput* renderOutput);

	//based on the gjk status to decide whether we should do full contact gen with GJK/EPA normal. Also, this method store
	//GJK/EPA point to the manifold in case full contact gen doesn't generate any contact
	EHDV bool addGJKEPAContacts(const Gu::GjkConvex* relativeConvex, const Gu::GjkConvex* localConvex, const aos::PxMatTransformV& aToB, Gu::GjkStatus status,
		Gu::PersistentContact* manifoldContacts, const aos::FloatV replaceBreakingThreshold, const aos::FloatV toleranceLength, GjkOutput& output,
		Gu::PersistentContactManifold& manifold);

	//MTD code for box/convexhull vs box/convexhull
	EHDV bool computeMTD(const Gu::PolygonalData& polyData0, const Gu::PolygonalData& polyData1, const SupportLocal* map0, const SupportLocal* map1, aos::FloatV& penDepth, aos::Vec3V& normal);
	
	//MTD code for capsule vs box/convexhull
	EHDV bool computeMTD(const Gu::CapsuleV& capsule, const Gu::PolygonalData& polyData, const Gu::SupportLocal* map, aos::FloatV& penDepth, aos::Vec3V& normal);

	//full contact gen code for sphere vs convexhull
	EHDV bool generateSphereFullContactManifold(const Gu::CapsuleV& capsule, const Gu::PolygonalData& polyData, const Gu::SupportLocal* map, Gu::PersistentContact* manifoldContacts, PxU32& numContacts,
		const aos::FloatVArg contactDist, aos::Vec3V& normal, bool doOverlapTest);
}  
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactGenBoxConvex.cpp
#define PCM_USE_INTERNAL_OBJECT 1

using namespace px;
using namespace Gu;
using namespace aos;

//Precompute the convex data
//     7+------+6			0 = ---
//     /|     /|			1 = +--
//    / |    / |			2 = ++-
//   / 4+---/--+5			3 = -+-
// 3+------+2 /    y   z	4 = --+
//  | /    | /     |  /		5 = +-+
//  |/     |/      |/		6 = +++
// 0+------+1      *---x	7 = -++

EHDI static bool testFaceNormal(const PolygonalData& polyData0, const PolygonalData& polyData1, const SupportLocal* map0, const SupportLocal* map1, const PxMatTransformV& transform0To1, const PxMatTransformV& transform1To0, const FloatVArg contactDist, 
	FloatV& minOverlap, PxU32& feature, Vec3V& faceNormal, FeatureStatus faceStatus, FeatureStatus& status)
{
	E_UNUSED(polyData1);
		
	FloatV _minOverlap = FMax();//minOverlap;
	PxU32 _feature = 0;
	Vec3V _faceNormal = faceNormal;
	FloatV min0, max0;
	FloatV min1, max1;
	
	const Vec3V center1To0 = transform1To0.p;
		
	const Vec3V zeroV = V3Zero();
	const Vec3V shapeSpaceCenter1 = V3LoadU(polyData1.mCenter);
	const Vec3V internalCenter1In0 = transform1To0.transform(shapeSpaceCenter1);
	const FloatV internalRadius1 = FLoad(polyData1.mInternal.mInternalRadius);
	const Vec3V internalExtents1 = V3LoadU_SafeReadW(polyData1.mInternal.mInternalExtents);
	const Vec3V negInternalExtents1 = V3Neg(internalExtents1);

	//in the local space of polyData0
	for(PxU32 i=0; i<polyData0.mNbPolygons; ++i)
	{
		const HullPolygonData& polygon = polyData0.mPolygons[i];

		const Vec3V minVert = V3LoadU(polyData0.mVerts[polygon.mMinIndex]);
		const FloatV planeDist = FLoad(polygon.mPlane.d);
		const Vec3V vertexSpacePlaneNormal = V3LoadU(polygon.mPlane.n);

		//transform plane n to shape space
		const Vec3V shapeSpacePlaneNormal = M33TrnspsMulV3(map0->shape2Vertex, vertexSpacePlaneNormal);

		const FloatV magnitude = FRecip(V3Length(shapeSpacePlaneNormal));
		//ML::use this to avoid LHS
		min0 = FMul(V3Dot(vertexSpacePlaneNormal, minVert), magnitude);
		max0 = FMul(FNeg(planeDist), magnitude);

		//normalize shape space normal
		const Vec3V n0 = V3Scale(shapeSpacePlaneNormal, magnitude);

		//calculate polyData1 projection
		//rotate polygon's normal into the local space of polyData1
		const Vec3V n1 = transform0To1.rotate(n0);

		//test internal object	
		//ML: we don't need to transform the normal into the vertex space. If polyData1 don't have scale,
		//the vertex2Shape matrix will be identity, shape space normal will be the same as vertex space's normal.
		//If polyData0 have scale, internalExtens1 will be 0.
		const Vec3V proj = V3Sel(V3IsGrtr(n1, zeroV), internalExtents1, negInternalExtents1);
		const FloatV radius = FMax(V3Dot(n1, proj), internalRadius1);
		const FloatV internalTrans = V3Dot(internalCenter1In0, n0);

		const FloatV _min1 = FSub(internalTrans, radius);
		const FloatV _max1 = FAdd(internalTrans, radius);

		const FloatV _min = FMax(min0, _min1);
		const FloatV _max = FMin(max0, _max1);

		const FloatV _tempOverlap = FSub(_max, _min);
		//const FloatV _tempOverlap = FSub(max0, _min1);	

		//Internal object overlaps more than current min, so can skip it
		//because (a) it isn't a separating axis and (b) it isn't the smallest axis
		if(FAllGrtr(_tempOverlap, _minOverlap))
			continue;
		const FloatV translate = V3Dot(center1To0, n0);
		map1->doSupport(n1, min1, max1);

		min1 = FAdd(translate, min1);
		max1 = FAdd(translate, max1);

		const BoolV con = BOr(FIsGrtr(min1, FAdd(max0, contactDist)), FIsGrtr(min0, FAdd(max1, contactDist)));

		if(BAllEqTTTT(con))
			return false;

		const FloatV tempOverlap = FSub(max0, min1);

		if(FAllGrtr(_minOverlap, tempOverlap))
		{
			_minOverlap = tempOverlap;
			_feature = i;
			_faceNormal = n0;
		}
	}   

	if(FAllGrtr(minOverlap, _minOverlap))
	{
		faceNormal = _faceNormal; 
		minOverlap = _minOverlap;
		status = faceStatus;
	}

	feature = _feature;

	return true;
}

//plane is in the shape space of polyData
EHDI static void buildPartialHull(const PolygonalData& polyData, const SupportLocal* map, SeparatingAxes& validAxes, const Vec3VArg planeP, const Vec3VArg planeDir)
{		
	const FloatV zero = FZero();
	const Vec3V dir = V3Normalize(planeDir);
	for(PxU32 i=0; i<polyData.mNbPolygons; ++i)
	{
		const HullPolygonData& polygon = polyData.mPolygons[i];
		const PxU8* inds = polyData.mPolygonVertexRefs + polygon.mVRef8;
			
		Vec3V v0 = M33MulV3(map->vertex2Shape, V3LoadU_SafeReadW(polyData.mVerts[inds[polygon.mNbVerts - 1]]));	// PT: safe because of the way vertex memory is allocated in ConvexHullData
		
		FloatV dist0 = V3Dot(dir, V3Sub(v0, planeP));

		for (PxU32 iStart = 0; iStart < polygon.mNbVerts; iStart++)
		{
			const Vec3V v1 = M33MulV3(map->vertex2Shape, V3LoadU_SafeReadW(polyData.mVerts[inds[iStart]]));	// PT: safe because of the way vertex memory is allocated in ConvexHullData
				
			const FloatV dist1 = V3Dot(dir, V3Sub(v1, planeP));

			const BoolV con = BOr(FIsGrtr(dist0, zero), FIsGrtr(dist1, zero));

			//cull edge if either of the vertex will on the positive size of the plane
			if(BAllEqTTTT(con))
			{
				const Vec3V tempV = V3Sub(v0, v1); 
				PxVec3 temp;
				V3StoreU(tempV, temp);
				validAxes.addAxis(temp.getNormalized());
			}

			v0 = v1;
			dist0 = dist1;
		}
	}
}

EHDI static bool testEdgeNormal(const PolygonalData& polyData0, const PolygonalData& polyData1, const SupportLocal* map0, const SupportLocal* map1, const PxMatTransformV& transform0To1, const PxMatTransformV& transform1To0, const FloatVArg contactDist,
	FloatV& minOverlap, Vec3V& edgeNormalIn0, FeatureStatus edgeStatus, FeatureStatus& status)
{
	FloatV overlap = minOverlap;
	FloatV min0, max0;
	FloatV min1, max1;
	const FloatV eps = FEps();

	const Vec3V shapeSpaceCenter0 = V3LoadU(polyData0.mCenter);
	const Vec3V shapeSpaceCenter1 = V3LoadU(polyData1.mCenter);

	const Vec3V zeroV = V3Zero();
	const Vec3V internalCenter1In0 = V3Sub(transform1To0.transform(shapeSpaceCenter1), shapeSpaceCenter0);
	const FloatV internalRadius1 = FLoad(polyData1.mInternal.mInternalRadius);
	const Vec3V internalExtents1 = V3LoadU_SafeReadW(polyData1.mInternal.mInternalExtents);
	const Vec3V negInternalExtents1 = V3Neg(internalExtents1);

	const FloatV internalRadius0 = FLoad(polyData0.mInternal.mInternalRadius);
	const Vec3V internalExtents0 = V3LoadU_SafeReadW(polyData0.mInternal.mInternalExtents);
	const Vec3V negInternalExtents0 = V3Neg(internalExtents0);

	const Vec3V center1To0 = transform1To0.p;

	//in polyData0 shape space
	const Vec3V dir0 = V3Sub(transform1To0.transform(shapeSpaceCenter1), shapeSpaceCenter0);
	const Vec3V support0 = map0->doSupport(dir0);

	//in polyData1 shape space
	const Vec3V dir1 = transform0To1.rotate(V3Neg(dir0));
	const Vec3V support1 = map1->doSupport(dir1);
		
	const Vec3V support0In1 = transform0To1.transform(support0);
	const Vec3V support1In0 = transform1To0.transform(support1);

	SeparatingAxes mSA0;
	SeparatingAxes mSA1;
	mSA0.reset();
	mSA1.reset();
		
	buildPartialHull(polyData0, map0, mSA0, support1In0, dir0);
	buildPartialHull(polyData1, map1, mSA1, support0In1, dir1);

	const PxVec3* axe0 = mSA0.getAxes();
	const PxVec3* axe1 = mSA1.getAxes();
	const PxU32 numAxe0 = mSA0.getNumAxes();
	const PxU32 numAxe1 = mSA1.getNumAxes();
	for(PxU32 i=0; i < numAxe0; ++i)
	{
		//axe0[i] is in the shape space of polyData0
		const Vec3V v0 = V3LoadU(axe0[i]);

		for(PxU32 j=0; j< numAxe1; ++j)
		{
			//axe1[j] is in the shape space of polyData1
			const Vec3V v1 = V3LoadU(axe1[j]);

			const Vec3V dir = V3Cross(v0, transform1To0.rotate(v1));
			const FloatV lenSq = V3Dot(dir, dir);
			if(FAllGrtr(eps, lenSq))
				continue;

			//n0 is in polyData0's local space
			const Vec3V n0 = V3Scale(dir, FRsqrt(lenSq));
				
			//n1 is in polyData1's local space
			const Vec3V n1 = transform0To1.rotate(n0);

			//ML: we don't need to transform the normal into the vertex space. If polyData1 don't have scale,
			//the vertex2Shape matrix will be identity, shape space normal will be the same as vertex space's normal.
			//If polyData0 have scale, internalExtens1 will be 0.
			//vertex space n1
			const Vec3V proj = V3Sel(V3IsGrtr(n1, zeroV), internalExtents1, negInternalExtents1);
			const FloatV radius = FMax(V3Dot(n1, proj), internalRadius1);

			const FloatV internalTrans = V3Dot(internalCenter1In0, n0);

			const FloatV _min1 = FSub(internalTrans, radius);
			const FloatV _max1 = FAdd(internalTrans, radius);

			const Vec3V proj0 = V3Sel(V3IsGrtr(n0, zeroV), internalExtents0, negInternalExtents0);
			const FloatV radius0 = FMax(V3Dot(n0, proj0), internalRadius0);

			const FloatV _max0 = radius0;
			const FloatV _min0 = FNeg(radius0);

			E_ASSERT(FAllGrtrOrEq(_max0, _min0));
			E_ASSERT(FAllGrtrOrEq(_max1, _min1));

			const FloatV _min = FMax(_min0, _min1);
			const FloatV _max = FMin(_max0, _max1);

			const FloatV _tempOverlap = FSub(_max, _min);

			//Internal object overlaps more than current min, so can skip it
			//because (a) it isn't a separating axis and (b) it isn't the smallest axis
			if(FAllGrtr(_tempOverlap, overlap))
				continue;
			//get polyData0's projection
			map0->doSupport(n0, min0, max0);

			const FloatV translate = V3Dot(center1To0, n0);

			//get polyData1's projection
			map1->doSupport(n1, min1, max1);

			min1 = FAdd(translate, min1);
			max1 = FAdd(translate, max1);

			const BoolV con = BOr(FIsGrtr(min1, FAdd(max0, contactDist)), FIsGrtr(min0, FAdd(max1, contactDist)));
			if(BAllEqTTTT(con))
				return false;

			const FloatV tempOverlap = FSub(max0, min1);

			E_ASSERT(FAllGrtrOrEq(tempOverlap, _tempOverlap));
			if(FAllGrtr(overlap, tempOverlap))
			{
				overlap = tempOverlap;
				edgeNormalIn0 = n0;
				status = edgeStatus;
			}
		}
	}

	minOverlap = overlap;
		
	return true;
}

//contactNormal is in the space of polyData0
EHDI static void generatedContacts(const PolygonalData& polyData0, const PolygonalData& polyData1, const HullPolygonData& referencePolygon, const HullPolygonData& incidentPolygon,
	const SupportLocal* map0, const SupportLocal* map1, const PxMatTransformV& transform0To1, PersistentContact* manifoldContacts, 
	PxU32& numContacts, const FloatVArg contactDist, PxRenderOutput* renderOutput)
{
	E_UNUSED(renderOutput);
	const FloatV zero = FZero();

	const PxU8* inds0 = polyData0.mPolygonVertexRefs + referencePolygon.mVRef8;

	//transform the plane normal to shape space
	const Vec3V contactNormal = V3Normalize(M33TrnspsMulV3(map0->shape2Vertex, V3LoadU(referencePolygon.mPlane.n)));
		
	//this is the matrix transform all points to the 2d plane
	const Mat33V rot = findRotationMatrixFromZAxis(contactNormal);

	const PxU8* inds1 = polyData1.mPolygonVertexRefs + incidentPolygon.mVRef8;

	Vec3V points0In0_buf[E_ALLOCA_N]; Vec3V* points0In0 = points0In0_buf;
	Vec3V points1In0_buf[E_ALLOCA_N]; Vec3V* points1In0 = points1In0_buf;
	bool points1In0Penetration_buf[E_ALLOCA_N]; bool* points1In0Penetration = points1In0Penetration_buf;
	FloatV points1In0TValue_buf[E_ALLOCA_N]; FloatV* points1In0TValue = points1In0TValue_buf;

	//Transform all the verts from vertex space to shape space
	map0->populateVerts(inds0, referencePolygon.mNbVerts, polyData0.mVerts, points0In0);
	map1->populateVerts(inds1, incidentPolygon.mNbVerts, polyData1.mVerts, points1In0);

	
	//This is used to calculate the project point when the 2D reference face points is inside the 2D incident face point
	const Vec3V sPoint = points1In0[0]; 
	
	E_ASSERT(incidentPolygon.mNbVerts <= 64);
 
	Vec3V eps = Vec3V_From_FloatV(FEps());
	Vec3V max = Vec3V_From_FloatV(FMax());
	Vec3V nmax = V3Neg(max); 

	//transform reference polygon to 2d, calculate min and max
	Vec3V rPolygonMin = max;
	Vec3V rPolygonMax = nmax;
	for(PxU32 i=0; i<referencePolygon.mNbVerts; ++i)
	{
		points0In0[i] = M33MulV3(rot, points0In0[i]);
		rPolygonMin = V3Min(rPolygonMin, points0In0[i]);
		rPolygonMax = V3Max(rPolygonMax, points0In0[i]);
	}
		
	rPolygonMin = V3Sub(rPolygonMin, eps);
	rPolygonMax = V3Add(rPolygonMax, eps);

	const FloatV d = V3GetZ(points0In0[0]);
	const FloatV rd = FAdd(d, contactDist);

	Vec3V iPolygonMin = max; 
	Vec3V iPolygonMax = nmax;

	PxU32 inside = 0;
	for(PxU32 i=0; i<incidentPolygon.mNbVerts; ++i)
	{
		const Vec3V vert1 = points1In0[i]; //this still in polyData1's local space
		const Vec3V a = transform0To1.transformInv(vert1);
		points1In0[i] = M33MulV3(rot, a);
		const FloatV z = V3GetZ(points1In0[i]);
		points1In0TValue[i] = FSub(z, d);
		points1In0[i] = V3SetZ(points1In0[i], d);
		iPolygonMin = V3Min(iPolygonMin, points1In0[i]);
		iPolygonMax = V3Max(iPolygonMax, points1In0[i]);
		if(FAllGrtr(rd, z))
		{
			points1In0Penetration[i] = true;

			if(contains(points0In0, referencePolygon.mNbVerts, points1In0[i], rPolygonMin, rPolygonMax))
			{
				inside++;

				if (numContacts == PxContactBuffer::MAX_CONTACTS)
					return;
				
				const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(contactNormal), points1In0TValue[i]);
				manifoldContacts[numContacts].mLocalPointA = vert1;
				manifoldContacts[numContacts].mLocalPointB = M33TrnspsMulV3(rot, points1In0[i]);
				manifoldContacts[numContacts++].mLocalNormalPen = localNormalPen;
			}
		}
		else
		{
			points1In0Penetration[i] = false;
		}
	}

	if(inside == incidentPolygon.mNbVerts)
		return;

	inside = 0;
	iPolygonMin = V3Sub(iPolygonMin, eps);
	iPolygonMax = V3Add(iPolygonMax, eps);

	const Vec3V incidentNormal = V3Normalize(M33TrnspsMulV3(map1->shape2Vertex, V3LoadU(incidentPolygon.mPlane.n)));
	
	const Vec3V contactNormalIn1 = transform0To1.rotate(contactNormal);

	for(PxU32 i=0; i<referencePolygon.mNbVerts; ++i)
	{
		if(contains(points1In0, incidentPolygon.mNbVerts, points0In0[i], iPolygonMin, iPolygonMax))
		{
			//const Vec3V vert0=Vec3V_From_PxVec3(polyData0.mVerts[inds0[i]]);
			const Vec3V vert0 = M33TrnspsMulV3(rot, points0In0[i]);
			const Vec3V a = transform0To1.transform(vert0);

			const FloatV nom = V3Dot(incidentNormal, V3Sub(sPoint, a)); 
			const FloatV denom = V3Dot(incidentNormal, contactNormalIn1);
			E_ASSERT(FAllEq(denom, zero)==0);
			const FloatV t = FDiv(nom, denom);

			if(FAllGrtr(t, contactDist))
				continue;

			inside++;

			if (numContacts == PxContactBuffer::MAX_CONTACTS)
				return;
			
			const Vec3V projPoint = V3ScaleAdd(contactNormalIn1, t, a);
			const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(contactNormal), t);

			manifoldContacts[numContacts].mLocalPointA = projPoint;
			manifoldContacts[numContacts].mLocalPointB = vert0;
			manifoldContacts[numContacts++].mLocalNormalPen = localNormalPen;
		}
	}

	if(inside == referencePolygon.mNbVerts)
		return;

	//(2) segment intersection
	for (PxU32 iStart = 0, iEnd = PxU32(incidentPolygon.mNbVerts - 1); iStart < incidentPolygon.mNbVerts; iEnd = iStart++)
	{
		if((!points1In0Penetration[iStart] && !points1In0Penetration[iEnd] ) )//|| (points1In0[i].status == POINT_OUTSIDE && points1In0[incidentIndex].status == POINT_OUTSIDE))
			continue;

		const Vec3V ipA = points1In0[iStart];
		const Vec3V ipB = points1In0[iEnd];

		Vec3V ipAOri = V3SetZ(points1In0[iStart], FAdd(points1In0TValue[iStart], d));
		Vec3V ipBOri = V3SetZ(points1In0[iEnd], FAdd(points1In0TValue[iEnd], d));

		const Vec3V iMin = V3Min(ipA, ipB);
		const Vec3V iMax = V3Max(ipA, ipB);

		for (PxU32 rStart = 0, rEnd = PxU32(referencePolygon.mNbVerts - 1); rStart < referencePolygon.mNbVerts; rEnd = rStart++) 
		{
			const Vec3V rpA = points0In0[rStart];
			const Vec3V rpB = points0In0[rEnd];

			const Vec3V rMin = V3Min(rpA, rpB);
			const Vec3V rMax = V3Max(rpA, rpB);
				
			const BoolV tempCon = BOr(V3IsGrtr(iMin, rMax), V3IsGrtr(rMin, iMax));
			const BoolV con = BOr(BGetX(tempCon), BGetY(tempCon));
		
			if(BAllEqTTTT(con))
				continue;
					
			FloatV a1 = signed2DTriArea(rpA, rpB, ipA);
			FloatV a2 = signed2DTriArea(rpA, rpB, ipB);

			if(FAllGrtr(zero, FMul(a1, a2)))
			{
				FloatV a3 = signed2DTriArea(ipA, ipB, rpA);
				FloatV a4 = signed2DTriArea(ipA, ipB, rpB);

				if(FAllGrtr(zero, FMul(a3, a4)))
				{
					//these two segment intersect

					const FloatV t = FDiv(a1, FSub(a2, a1));

					const Vec3V pBB = V3NegScaleSub(V3Sub(ipBOri, ipAOri), t, ipAOri); 
					const Vec3V pAA = V3SetZ(pBB, d);
					const Vec3V pA = M33TrnspsMulV3(rot, pAA);
					const Vec3V pB = transform0To1.transform(M33TrnspsMulV3(rot, pBB));
					const FloatV pen = FSub(V3GetZ(pBB), V3GetZ(pAA));
					if(FAllGrtr(pen, contactDist))
						continue;

					if (numContacts == PxContactBuffer::MAX_CONTACTS)
						return;

					const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(contactNormal), pen);
					manifoldContacts[numContacts].mLocalPointA = pB;
					manifoldContacts[numContacts].mLocalPointB = pA;
					manifoldContacts[numContacts++].mLocalNormalPen = localNormalPen;
				}
			}
		}
	}
}

EHDI bool Gu::generateFullContactManifold(const PolygonalData& polyData0, const PolygonalData& polyData1, const SupportLocal* map0, const SupportLocal* map1, PersistentContact* manifoldContacts, PxU32& numContacts,
	const FloatVArg contactDist, const Vec3VArg normal, const Vec3VArg closestA, const Vec3VArg closestB, PxReal marginA, PxReal marginB, bool doOverlapTest, 
	PxRenderOutput* renderOutput, PxReal toleranceLength)
{
	const PxMatTransformV transform1To0V = map0->transform.transformInv(map1->transform);
	const PxMatTransformV transform0To1V = map1->transform.transformInv(map0->transform);
	
	if(doOverlapTest)
	{
		//if gjk fail, SAT based yes/no test
		FeatureStatus status = POLYDATA0;
		FloatV minOverlap = FMax();
		Vec3V minNormal = V3Zero();
	
		PxU32 feature0;
		//in the local space of polyData0, minNormal is in polyData0 space
		if(!testFaceNormal(polyData0, polyData1, map0, map1, transform0To1V, transform1To0V, contactDist, minOverlap, feature0, minNormal, POLYDATA0, status))
			return false;
			
		PxU32 feature1;
		//in the local space of polyData1, if minOverlap is overwrite inside this function, minNormal will be in polyData1 space
		if(!testFaceNormal(polyData1, polyData0, map1, map0, transform1To0V, transform0To1V, contactDist, minOverlap, feature1, minNormal, POLYDATA1, status))
			return false;

		bool doEdgeTest = false;
			
EdgeTest:
		if(doEdgeTest)
		{
			if(!testEdgeNormal(polyData0, polyData1, map0, map1, transform0To1V, transform1To0V, contactDist, minOverlap, minNormal, EDGE, status))
				return false;

			if(status != EDGE)
				return true;
		}
			
		if(status == POLYDATA0)
		{
			//minNormal is in the local space of polydata0
		
			const HullPolygonData& referencePolygon = polyData0.mPolygons[feature0];
			const Vec3V n = transform0To1V.rotate(minNormal);
			const HullPolygonData& incidentPolygon = polyData1.mPolygons[getPolygonIndex(polyData1, map1, n)];
				
			generatedContacts(polyData0, polyData1, referencePolygon, incidentPolygon, map0, map1, transform0To1V, manifoldContacts, numContacts, contactDist, renderOutput);
				
			if (numContacts > 0)
			{
				const Vec3V nn = V3Neg(n);
				//flip the contacts
				for(PxU32 i=0; i<numContacts; ++i)
				{
					const Vec3V localPointB = manifoldContacts[i].mLocalPointB;
					manifoldContacts[i].mLocalPointB = manifoldContacts[i].mLocalPointA;
					manifoldContacts[i].mLocalPointA = localPointB;
					manifoldContacts[i].mLocalNormalPen = V4SetW(nn, V4GetW(manifoldContacts[i].mLocalNormalPen));
				}
			}
		}
		else if(status == POLYDATA1)
		{
			//minNormal is in the local space of polydata1
			const HullPolygonData& referencePolygon = polyData1.mPolygons[feature1];
			const HullPolygonData& incidentPolygon = polyData0.mPolygons[getPolygonIndex(polyData0, map0, transform1To0V.rotate(minNormal))];
				
			//reference face is polyData1
			generatedContacts(polyData1, polyData0, referencePolygon, incidentPolygon, map1, map0, transform1To0V, manifoldContacts, numContacts, contactDist, renderOutput);

		}
		else //if(status == EDGE0)
		{
			//minNormal is in the local space of polydata0
			
			const HullPolygonData& incidentPolygon = polyData0.mPolygons[getPolygonIndex(polyData0, map0, V3Neg(minNormal))];
			const HullPolygonData& referencePolygon = polyData1.mPolygons[getPolygonIndex(polyData1, map1, transform0To1V.rotate(minNormal))];
			generatedContacts(polyData1, polyData0, referencePolygon, incidentPolygon, map1, map0, transform1To0V, manifoldContacts, numContacts, contactDist, renderOutput);
		}

		if(numContacts == 0 && !doEdgeTest)
		{
			doEdgeTest = true;
			goto EdgeTest;
		}
	}
	else
	{
		const PxReal lowerEps = toleranceLength * PCM_WITNESS_POINT_LOWER_EPS;
		const PxReal upperEps = toleranceLength * PCM_WITNESS_POINT_UPPER_EPS;
		const PxReal toleranceA = PxClamp(marginA, lowerEps, upperEps);
		const PxReal toleranceB = PxClamp(marginB, lowerEps, upperEps);

		const Vec3V negNormal = V3Neg(normal);
		const Vec3V normalIn0 = transform0To1V.rotateInv(normal);

		PxU32 faceIndex1 = getWitnessPolygonIndex(polyData1, map1, negNormal, closestB, toleranceB);
		PxU32 faceIndex0 = getWitnessPolygonIndex(polyData0, map0, normalIn0, transform0To1V.transformInv(closestA),
			toleranceA);

		const HullPolygonData& referencePolygon = polyData1.mPolygons[faceIndex1];
		const HullPolygonData& incidentPolygon = polyData0.mPolygons[faceIndex0];

		const Vec3V referenceNormal = V3Normalize(M33TrnspsMulV3(map1->shape2Vertex, V3LoadU(referencePolygon.mPlane.n)));
		const Vec3V incidentNormal = V3Normalize(M33TrnspsMulV3(map0->shape2Vertex, V3LoadU(incidentPolygon.mPlane.n)));
		
		const FloatV referenceProject = FAbs(V3Dot(referenceNormal, negNormal));
		const FloatV incidentProject = FAbs(V3Dot(incidentNormal, normalIn0));

		if (FAllGrtrOrEq(referenceProject, incidentProject))
		{
			generatedContacts(polyData1, polyData0, referencePolygon, incidentPolygon, map1, map0, transform1To0V, manifoldContacts, numContacts, contactDist, renderOutput);
		}
		else
		{
			generatedContacts(polyData0, polyData1, incidentPolygon, referencePolygon, map0, map1, transform0To1V, manifoldContacts, numContacts, contactDist, renderOutput);

			if (numContacts > 0)
			{
				const Vec3V n = transform0To1V.rotate(incidentNormal);

				const Vec3V nn = V3Neg(n);
				//flip the contacts
				for (PxU32 i = 0; i<numContacts; ++i)
				{
					const Vec3V localPointB = manifoldContacts[i].mLocalPointB;
					manifoldContacts[i].mLocalPointB = manifoldContacts[i].mLocalPointA;
					manifoldContacts[i].mLocalPointA = localPointB;
					manifoldContacts[i].mLocalNormalPen = V4SetW(nn, V4GetW(manifoldContacts[i].mLocalNormalPen));
				}
			}
		}
	}
	
	return true;
}

//This function called by box/convex and convex/convex pcm function
EHDI bool Gu::addGJKEPAContacts(const GjkConvex* relativeConvex, const GjkConvex* localConvex, const PxMatTransformV& aToB, GjkStatus status,
	PersistentContact* manifoldContacts, const FloatV replaceBreakingThreshold, const FloatV tolerenceLength, GjkOutput& output,
	PersistentContactManifold& manifold)
{
	bool doOverlapTest = false;
	if (status == GJK_DEGENERATE)
	{
		const Vec3V normal = output.normal;
		const FloatV costheta = V3Dot(output.searchDir, normal);
		if (FAllGrtr(costheta, FLoad(0.9999f)))
		{
			const Vec3V centreA = relativeConvex->getCenter();
			const Vec3V centreB = localConvex->getCenter();

			const Vec3V dir = V3Normalize(V3Sub(centreA, centreB));
			const FloatV theta = V3Dot(dir, normal);

			if (FAllGrtr(theta, FLoad(0.707f)))
			{
				addManifoldPoint(manifoldContacts, manifold, output, aToB, replaceBreakingThreshold);
			}
			else
			{
				doOverlapTest = true;
			}
		}
		else
		{
			doOverlapTest = true;
		}
	}
	else if (status == GJK_CONTACT)
	{
		addManifoldPoint(manifoldContacts, manifold, output, aToB, replaceBreakingThreshold);
	}
	else
	{
		E_ASSERT(status == EPA_CONTACT);

		status = epaPenetration(*relativeConvex, *localConvex, manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints,
			true, tolerenceLength, output);
		if (status == EPA_CONTACT)
		{
			addManifoldPoint(manifoldContacts, manifold, output, aToB, replaceBreakingThreshold);
		}
		else
		{
			doOverlapTest = true;
		}
	}

	return doOverlapTest;
}

EHDI bool Gu::computeMTD(const PolygonalData& polyData0, const PolygonalData& polyData1, const SupportLocal* map0, const SupportLocal* map1, FloatV& penDepth, Vec3V& normal)
{
	using namespace aos;

	const PxMatTransformV transform1To0V = map0->transform.transformInv(map1->transform);
	const PxMatTransformV transform0To1V = map1->transform.transformInv(map0->transform);
	
	FeatureStatus status = POLYDATA0;
	FloatV minOverlap = FMax();
	Vec3V minNormal = V3Zero();
	const FloatV contactDist = FZero();

	PxU32 feature0;
	//in the local space of polyData0, minNormal is in polyData0 space
	if(!testFaceNormal(polyData0, polyData1, map0, map1, transform0To1V, transform1To0V, contactDist, minOverlap, feature0, minNormal, POLYDATA0, status))
		return false;
		
	PxU32 feature1;
	//in the local space of polyData1, if minOverlap is overwrite inside this function, minNormal will be in polyData1 space
	if(!testFaceNormal(polyData1, polyData0, map1, map0, transform1To0V, transform0To1V, contactDist, minOverlap, feature1, minNormal, POLYDATA1, status))
		return false;
	
	if(!testEdgeNormal(polyData0, polyData1, map0, map1, transform0To1V, transform1To0V, contactDist, minOverlap, minNormal, EDGE, status))
		return false;

	penDepth = minOverlap;
	if(status == POLYDATA1)
	{
		//minNormal is in the local space of polydata1
		normal = map1->transform.rotate(minNormal);
	}
	else
	{
		E_ASSERT(status == POLYDATA0 || status == EDGE);
		//ML: status == POLYDATA0 or status == EDGE, minNormal is in the local space of polydata0
		normal = V3Neg(map0->transform.rotate(minNormal));
	}

	return true;
}

// ===== 원본: physx/source/geomutils/src/contact/GuContactMethodImpl.h
namespace px
{
	class PxGeometry;
	class PxRenderOutput;
	class PxContactBuffer;

namespace Gu
{
	class PersistentContactManifold;
	class MultiplePersistentContactManifold;

	struct NarrowPhaseParams
	{
		EHD	NarrowPhaseParams(PxReal contactDistance, PxReal meshContactMargin, PxReal toleranceLength) :
				mContactDistance(contactDistance),
				mMeshContactMargin(meshContactMargin),
				mToleranceLength(toleranceLength)	{}

		PxReal			mContactDistance;
		const PxReal	mMeshContactMargin;	// PT: Margin used to generate mesh contacts. Temp & unclear, should be removed once GJK is default path.
		const PxReal	mToleranceLength;	// PT: copy of PxTolerancesScale::length
	};

	enum ManifoldFlags
	{
		IS_MANIFOLD			= (1<<0),
		IS_MULTI_MANIFOLD	= (1<<1)
	};

	struct Cache : public PxCache
	{
		EHDI Cache()
		{
		}

		EHD void setManifold(void* manifold)
		{
			E_ASSERT((size_t(manifold) & 0xF) == 0);
			mCachedData = reinterpret_cast<PxU8*>(manifold);
			mManifoldFlags |= IS_MANIFOLD;
		}

		EHD void setMultiManifold(void* manifold)
		{
			E_ASSERT((size_t(manifold) & 0xF) == 0);
			mCachedData = reinterpret_cast<PxU8*>(manifold);
			mManifoldFlags |= IS_MANIFOLD|IS_MULTI_MANIFOLD;
		}

		EHD PxU8 isManifold()	const
		{
			return PxU8(mManifoldFlags & IS_MANIFOLD);
		}

		EHD PxU8 isMultiManifold()	const
		{
			return PxU8(mManifoldFlags & IS_MULTI_MANIFOLD);
		}

		EHD PersistentContactManifold& getManifold()
		{
			E_ASSERT(isManifold());
			E_ASSERT(!isMultiManifold());
			E_ASSERT((uintptr_t(mCachedData) & 0xf) == 0);
			return *reinterpret_cast<PersistentContactManifold*>(mCachedData);
		}

		EHD MultiplePersistentContactManifold& getMultipleManifold()
		{
			E_ASSERT(isManifold());
			E_ASSERT(isMultiManifold());
			E_ASSERT((uintptr_t(mCachedData) & 0xf) == 0);
			return *reinterpret_cast<MultiplePersistentContactManifold*>(mCachedData);
		}
	};
}

template<class Geom> EHD const Geom& checkedCast(const PxGeometry& geom)
{
	checkType<Geom>(geom);
	return static_cast<const Geom&>(geom);
}

#define GU_CONTACT_METHOD_ARGS				\
	const PxGeometry& shape0,				\
	const PxGeometry& shape1,				\
	const PxTransform32& transform0,		\
	const PxTransform32& transform1,		\
	const Gu::NarrowPhaseParams& params,	\
	Gu::Cache& cache,						\
	PxContactBuffer& contactBuffer,			\
	PxRenderOutput* renderOutput

#define GU_CONTACT_METHOD_ARGS_UNUSED	\
	const PxGeometry&,					\
	const PxGeometry&,					\
	const PxTransform32&,				\
	const PxTransform32&,				\
	const Gu::NarrowPhaseParams&,		\
	Gu::Cache&,							\
	PxContactBuffer&,					\
	PxRenderOutput*

namespace Gu
{
	 EHDV bool contactSphereSphere(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactSphereCapsule(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactSphereBox(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactCapsuleCapsule(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactCapsuleBox(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactCapsuleConvex(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactBoxBox(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactBoxConvex(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactConvexConvex(GU_CONTACT_METHOD_ARGS);
	
	 EHDV bool contactSphereMesh(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactCapsuleMesh(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactBoxMesh(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactConvexMesh(GU_CONTACT_METHOD_ARGS);

	 EHDV bool contactSphereHeightfield(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactCapsuleHeightfield(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactBoxHeightfield(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactConvexHeightfield(GU_CONTACT_METHOD_ARGS);
	
	 EHDV bool contactSpherePlane(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactPlaneBox(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactPlaneCapsule(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactPlaneConvexCore(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactPlaneConvex(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactPlaneMesh(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactMeshMesh(GU_CONTACT_METHOD_ARGS);

	 EHDV bool contactConvexCoreConvex(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactConvexCoreTrimesh(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactConvexCoreHeightfield(GU_CONTACT_METHOD_ARGS);

	 EHDV bool contactCustomGeometryGeometry(GU_CONTACT_METHOD_ARGS);
	 EHDV bool contactGeometryCustomGeometry(GU_CONTACT_METHOD_ARGS);

	 EHDV bool pcmContactSphereMesh(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactCapsuleMesh(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactBoxMesh(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactConvexMesh(GU_CONTACT_METHOD_ARGS);
	
	 EHDV bool pcmContactSphereHeightField(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactCapsuleHeightField(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactBoxHeightField(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactConvexHeightField(GU_CONTACT_METHOD_ARGS);

	 EHDV bool pcmContactPlaneCapsule(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactPlaneBox(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactPlaneConvex(GU_CONTACT_METHOD_ARGS);

	 EHDV bool pcmContactSphereSphere(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactSpherePlane(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactSphereCapsule(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactSphereBox(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactSphereConvex(GU_CONTACT_METHOD_ARGS);
	
	 EHDV bool pcmContactCapsuleCapsule(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactCapsuleBox(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactCapsuleConvex(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactBoxBox(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactBoxConvex(GU_CONTACT_METHOD_ARGS);
	 EHDV bool pcmContactConvexConvex(GU_CONTACT_METHOD_ARGS);

	 EHDV bool pcmContactGeometryCustomGeometry(GU_CONTACT_METHOD_ARGS);
}
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactConvexConvex.cpp
using namespace px;
using namespace Gu;
using namespace aos;

EHDI static bool fullContactsGenerationConvexConvex(const GjkConvex* relativeConvex, const GjkConvex* localConvex, const PxTransformV& transf0, const PxTransformV& transf1, 
											   bool idtScale0, bool idtScale1, PersistentContact* manifoldContacts, PxContactBuffer& contactBuffer, 
											   PersistentContactManifold& manifold, Vec3VArg normal, const Vec3VArg closestA, const Vec3VArg closestB,
											   const FloatVArg contactDist, bool doOverlapTest, PxRenderOutput* renderOutput, PxReal toleranceLength)
{
	PolygonalData polyData0, polyData1;
	const ConvexHullV& convexHull0 = relativeConvex->getConvex<ConvexHullV>();
	const ConvexHullV& convexHull1 = localConvex->getConvex<ConvexHullV>();
	getPCMConvexData(convexHull0, idtScale0, polyData0);
	getPCMConvexData(convexHull1, idtScale1, polyData1);

	alignas(16) PxU8 buff0[sizeof(SupportLocalImpl<ConvexHullV>)];
	alignas(16) PxU8 buff1[sizeof(SupportLocalImpl<ConvexHullV>)];

	SupportLocal* map0 = (idtScale0 ? static_cast<SupportLocal*>(E_PLACEMENT_NEW(buff0, SupportLocalImpl<ConvexHullNoScaleV>)(static_cast<const ConvexHullNoScaleV&>(convexHull0), transf0, convexHull0.vertex2Shape, convexHull0.shape2Vertex, idtScale0)) : 
		static_cast<SupportLocal*>(E_PLACEMENT_NEW(buff0, SupportLocalImpl<ConvexHullV>)(convexHull0, transf0, convexHull0.vertex2Shape, convexHull0.shape2Vertex, idtScale0)));

	SupportLocal* map1 = (idtScale1 ? static_cast<SupportLocal*>(E_PLACEMENT_NEW(buff1, SupportLocalImpl<ConvexHullNoScaleV>)(static_cast<const ConvexHullNoScaleV&>(convexHull1), transf1, convexHull1.vertex2Shape, convexHull1.shape2Vertex, idtScale1)) : 
		static_cast<SupportLocal*>(E_PLACEMENT_NEW(buff1, SupportLocalImpl<ConvexHullV>)(convexHull1, transf1, convexHull1.vertex2Shape, convexHull1.shape2Vertex, idtScale1)));

	PxU32 numContacts = 0;

	if(generateFullContactManifold(polyData0, polyData1, map0, map1, manifoldContacts, numContacts, contactDist, normal, closestA, closestB, convexHull0.getMarginF(), 
		convexHull1.getMarginF(), doOverlapTest, renderOutput, toleranceLength))
	{
		if(numContacts > 0)
		{
			//reduce contacts
			manifold.addBatchManifoldContacts(manifoldContacts, numContacts, toleranceLength);

			const Vec3V worldNormal = manifold.getWorldNormal(transf1);

			//add the manifold contacts;
			manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transf1, contactDist);

		}
		else
		{
			//if doOverlapTest is true, which means GJK/EPA degenerate so we won't have any contact in the manifoldContacts array
			if(!doOverlapTest)
			{
				const Vec3V worldNormal = manifold.getWorldNormal(transf1);

				manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transf1, contactDist);
			}
		}
		return true;
	}
	return false;
}

EHDI static bool generateOrProcessContactsConvexConvex(	const GjkConvex* relativeConvex, const GjkConvex* localConvex, const PxTransformV& transf0, const PxTransformV& transf1,
													const PxMatTransformV& aToB, GjkStatus status, GjkOutput& output, PersistentContactManifold& manifold, PxContactBuffer& contactBuffer,
													PxU32 initialContacts, const FloatV minMargin, const FloatV contactDist,
													bool idtScale0, bool idtScale1, PxReal toleranceLength, PxRenderOutput* renderOutput)
{
	if(status == GJK_NON_INTERSECT)
	{
		return false;
	}
	else
	{
		PersistentContact* manifoldContacts = EPX_CP_TO_PCP(contactBuffer.contacts);

		const Vec3V localNor = manifold.mNumContacts ? manifold.getLocalNormal() : V3Zero();

		const FloatV replaceBreakingThreshold = FMul(minMargin, FLoad(0.05f));

		//addGJKEPAContacts will increase the number of contacts in manifold. If status == EPA_CONTACT, we need to run epa algorithm and generate closest points, normal and
		//pentration. If epa doesn't degenerate, we will store the contacts information in the manifold. Otherwise, we will return true to do the fallback test
		const bool doOverlapTest = addGJKEPAContacts(relativeConvex, localConvex, aToB, status, manifoldContacts, replaceBreakingThreshold, FLoad(toleranceLength), output, manifold);
		
		//ML: after we refresh the contacts(newContacts) and generate a GJK/EPA contacts(we will store that in the manifold), if the number of contacts is still less than the original contacts,
		//which means we lose too mang contacts and we should regenerate all the contacts in the current configuration
		//Also, we need to look at the existing contacts, if the existing contacts has very different normal than the GJK/EPA contacts,
		//which means we should throw away the existing contacts and do full contact gen
		const bool fullContactGen = FAllGrtr(FLoad(0.707106781f), V3Dot(localNor, output.normal)) || (manifold.mNumContacts < initialContacts);

		if(fullContactGen || doOverlapTest)
		{
			return fullContactsGenerationConvexConvex(relativeConvex, localConvex, transf0, transf1, idtScale0, idtScale1, manifoldContacts, contactBuffer,
				manifold, output.normal, output.closestA, output.closestB, contactDist, doOverlapTest, renderOutput, toleranceLength);
		}
		else
		{
			const Vec3V newLocalNor = V3Add(localNor, output.normal);
			const Vec3V worldNormal = V3Normalize(transf1.rotate(newLocalNor));
			//const Vec3V worldNormal = transf1.rotate(normal);
			manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transf1, contactDist);
			return true;
		}
	}
}

EHDI static bool convexHullNoScale0(	const ConvexHullV& convexHull0, const ConvexHullV& convexHull1, const PxTransformV& transf0, const PxTransformV& transf1,
								const PxMatTransformV& aToB, GjkOutput& output, PersistentContactManifold& manifold, PxContactBuffer& contactBuffer,
								PxU32 initialContacts, const FloatV minMargin, const FloatV contactDist,
								bool idtScale1, PxReal toleranceLength, PxRenderOutput* renderOutput)
{
	const RelativeConvex<ConvexHullNoScaleV> convexA(static_cast<const ConvexHullNoScaleV&>(convexHull0), aToB);
	if(idtScale1)
	{
		const LocalConvex<ConvexHullNoScaleV> convexB(static_cast<const ConvexHullNoScaleV&>(convexHull1));
		GjkStatus status = gjkPenetration<RelativeConvex<ConvexHullNoScaleV>, LocalConvex<ConvexHullNoScaleV> >(convexA, convexB, aToB.p, contactDist, true,
						manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);

		return generateOrProcessContactsConvexConvex(&convexA, &convexB, transf0, transf1, aToB, status, output, manifold,
			contactBuffer, initialContacts, minMargin, contactDist, true, true, toleranceLength, renderOutput);
	}
	else
	{
		const LocalConvex<ConvexHullV> convexB(convexHull1);
		GjkStatus status = gjkPenetration<RelativeConvex<ConvexHullNoScaleV>, LocalConvex<ConvexHullV> >(convexA, convexB, aToB.p, contactDist, true,
					manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);

		return generateOrProcessContactsConvexConvex(&convexA, &convexB, transf0, transf1, aToB, status, output, manifold,
			contactBuffer, initialContacts, minMargin, contactDist, true, false, toleranceLength, renderOutput);
	}
}

EHDI static bool convexHullHasScale0(const ConvexHullV& convexHull0, const ConvexHullV& convexHull1, const PxTransformV& transf0, const PxTransformV& transf1,
								const PxMatTransformV& aToB, GjkOutput& output, PersistentContactManifold& manifold, PxContactBuffer& contactBuffer,
								PxU32 initialContacts, const FloatV minMargin, const FloatV contactDist,
								bool idtScale1, PxReal toleranceLength, PxRenderOutput* renderOutput)
{
	RelativeConvex<ConvexHullV> convexA(convexHull0, aToB);
	if(idtScale1)
	{
		const LocalConvex<ConvexHullNoScaleV> convexB(static_cast<const ConvexHullNoScaleV&>(convexHull1));
		GjkStatus status = gjkPenetration<RelativeConvex<ConvexHullV>, LocalConvex<ConvexHullNoScaleV> >(convexA, convexB, aToB.p, contactDist, true,
						manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints,output);
		
		return generateOrProcessContactsConvexConvex(&convexA, &convexB, transf0, transf1, aToB, status, output, manifold,
			contactBuffer, initialContacts, minMargin, contactDist, false, true, toleranceLength, renderOutput);
	}
	else
	{
		const LocalConvex<ConvexHullV> convexB(convexHull1);
		GjkStatus status = gjkPenetration<RelativeConvex<ConvexHullV>, LocalConvex<ConvexHullV> >(convexA, convexB, aToB.p, contactDist, true,
					manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);

		return generateOrProcessContactsConvexConvex(&convexA, &convexB, transf0, transf1, aToB, status, output, manifold,
			contactBuffer, initialContacts, minMargin, contactDist, false, false, toleranceLength, renderOutput);
	}
}

EHDI bool Gu::pcmContactConvexConvex(GU_CONTACT_METHOD_ARGS)
{
	const PxConvexMeshGeometry& shapeConvex0 = checkedCast<PxConvexMeshGeometry>(shape0);
	const PxConvexMeshGeometry& shapeConvex1 = checkedCast<PxConvexMeshGeometry>(shape1);

	PersistentContactManifold& manifold = cache.getManifold();

	const ConvexHullData* hullData0 = _getHullData(shapeConvex0);
	const ConvexHullData* hullData1 = _getHullData(shapeConvex1);
	PxPrefetchLine(hullData0);
	PxPrefetchLine(hullData1);

	E_ASSERT(transform1.q.isSane());
	E_ASSERT(transform0.q.isSane());

	const Vec3V vScale0 = V3LoadU_SafeReadW(shapeConvex0.scale.scale);	// PT: safe because 'rotation' follows 'scale' in PxMeshScale
	const Vec3V vScale1 = V3LoadU_SafeReadW(shapeConvex1.scale.scale);	// PT: safe because 'rotation' follows 'scale' in PxMeshScale
	const FloatV contactDist = FLoad(params.mContactDistance);

	//Transfer A into the local space of B
	const PxTransformV transf0 = loadTransformA(transform0);
	const PxTransformV transf1 = loadTransformA(transform1);
	const PxTransformV curRTrans(transf1.transformInv(transf0));
	const PxMatTransformV aToB(curRTrans);
	
	const PxReal toleranceLength = params.mToleranceLength;
	const FloatV convexMargin0 = CalculatePCMConvexMargin(hullData0, vScale0, toleranceLength);
	const FloatV convexMargin1 = CalculatePCMConvexMargin(hullData1, vScale1, toleranceLength);
	
	const PxU32 initialContacts = manifold.mNumContacts;

	const FloatV minMargin = FMin(convexMargin0, convexMargin1);
	const FloatV projectBreakingThreshold = FMul(minMargin, FLoad(0.8f));
	
	manifold.refreshContactPoints(aToB, projectBreakingThreshold, contactDist);

	//ML: after refreshContactPoints, we might lose some contacts
	const bool bLostContacts = (manifold.mNumContacts != initialContacts);

	const Vec3V extent0 = V3Mul(V3LoadU_SafeReadW(hullData0->mInternal.mInternalExtents), vScale0);
	const Vec3V extent1 = V3Mul(V3LoadU_SafeReadW(hullData1->mInternal.mInternalExtents), vScale1);

	const FloatV radiusA = V3Length(extent0);
	const FloatV radiusB = V3Length(extent1);

	if(bLostContacts || manifold.invalidate_BoxConvex(curRTrans, transf0.q, transf1.q, minMargin, radiusA, radiusB))
	{
		manifold.setRelativeTransform(curRTrans, transf0.q, transf1.q);
		
		const bool idtScale0 = shapeConvex0.scale.isIdentity();
		const bool idtScale1 = shapeConvex1.scale.isIdentity();
		const QuatV vQuat0 = QuatVLoadU(&shapeConvex0.scale.rotation.x);
		const QuatV vQuat1 = QuatVLoadU(&shapeConvex1.scale.rotation.x);
		
		// PT: safe loads because mCenterOfMass isn't the last data in the structure
		const ConvexHullV convexHull0(hullData0, V3LoadU_SafeReadW(hullData0->mCenterOfMass), vScale0, vQuat0, idtScale0);
		const ConvexHullV convexHull1(hullData1, V3LoadU_SafeReadW(hullData1->mCenterOfMass), vScale1, vQuat1, idtScale1);

		GjkOutput output;
		
		if(idtScale0)
		{
			return convexHullNoScale0(convexHull0, convexHull1, transf0, transf1, aToB, output, manifold,
				contactBuffer, initialContacts, minMargin, contactDist, idtScale1, toleranceLength, renderOutput);
		}
		else
		{
			return convexHullHasScale0(convexHull0, convexHull1, transf0, transf1, aToB, output, manifold,
				contactBuffer, initialContacts, minMargin, contactDist, idtScale1, toleranceLength, renderOutput);
		}
	}
	else if(manifold.getNumContacts()> 0)
	{
		const Vec3V worldNormal = manifold.getWorldNormal(transf1);
		manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transf1, contactDist);
		return true;
	}

	return false;
}

// ===== 원본: physx/source/geomutils/src/gjk/GuVecSphere.h
/**
\brief Represents a sphere defined by its center point and radius.
*/
namespace px
{
namespace Gu
{
	class SphereV : public ConvexV
	{
	public:
		/**
		\brief Constructor
		*/
		EHDI SphereV(): ConvexV(ConvexType::eSPHERE)
		{
			radius = aos::FZero();
			bMarginIsRadius = true;
		}

		EHDI SphereV(const aos::Vec3VArg _center, const aos::FloatV _radius) : ConvexV(ConvexType::eSPHERE, _center)
		{
			using namespace aos;
			radius = _radius;
			FStore(radius, &margin);
			FStore(radius, &minMargin);
			FStore(radius, &sweepMargin);
			bMarginIsRadius = true;
		}


		/**
		\brief Copy constructor
		*/
		EHDI SphereV(const SphereV& sphere) : ConvexV(ConvexType::eSPHERE), radius(sphere.radius)
		{

			margin = sphere.margin;
			minMargin = sphere.minMargin;
			sweepMargin = sphere.sweepMargin;
			bMarginIsRadius = true;
		}

		EHDI SphereV(const PxGeometry& geom) : ConvexV(ConvexType::eSPHERE, aos::V3Zero())
		{
			using namespace aos;
			const PxSphereGeometry& sphereGeom = static_cast<const PxSphereGeometry&>(geom);
			const FloatV r = FLoad(sphereGeom.radius);
			radius = r;
			margin = sphereGeom.radius;
			minMargin = sphereGeom.radius;
			sweepMargin = sphereGeom.radius;
			bMarginIsRadius = true;
		}

		/**
		\brief Destructor
		*/
		EHDI ~SphereV()
		{
		}

		EHDI	void	setV(const aos::Vec3VArg _center, const aos::FloatVArg _radius)		
		{ 
			center = _center;
			radius = _radius;
		}

		/**
		\brief Checks the sphere is valid.

		\return		true if the sphere is valid
		*/
		EHDI bool isValid() const
		{
			// Consistency condition for spheres: Radius >= 0.0f
			using namespace aos;
			return BAllEqTTTT(FIsGrtrOrEq(radius, FZero())) != 0;
		}

		/**
		\brief Tests if a point is contained within the sphere.

		\param[in] p the point to test
		\return	true if inside the sphere
		*/
		EHDI bool contains(const aos::Vec3VArg p) const
		{
			using namespace aos;
			const FloatV rr = FMul(radius, radius);
			const FloatV cc =  V3LengthSq(V3Sub(center, p));
			return FAllGrtrOrEq(rr, cc) != 0;
		}

		/**
		\brief Tests if a sphere is contained within the sphere.

		\param		sphere	[in] the sphere to test
		\return		true if inside the sphere
		*/
		EHDI bool contains(const SphereV& sphere)	const
		{
			using namespace aos;
			
			const Vec3V centerDif = V3Sub(center, sphere.center);
			const FloatV radiusDif = FSub(radius, sphere.radius);
			const FloatV cc = V3Dot(centerDif, centerDif);
			const FloatV rr = FMul(radiusDif, radiusDif); 

			const BoolV con0 = FIsGrtrOrEq(radiusDif, FZero());//might contain
			const BoolV con1 = FIsGrtr(rr, cc);//return true
			return BAllEqTTTT(BAnd(con0, con1))==1;
		}

		/**
		\brief Tests if a box is contained within the sphere.

		\param		minimum		[in] minimum value of the box
		\param		maximum		[in] maximum value of the box
		\return		true if inside the sphere
		*/
		EHDI bool contains(const aos::Vec3VArg minimum, const aos::Vec3VArg maximum) const
		{
		
			//compute the sphere which wrap around the box
			using namespace aos;
			const FloatV zero = FZero();
			const FloatV half = FHalf();

			const Vec3V boxSphereCenter = V3Scale(V3Add(maximum, minimum), half);
			const Vec3V v = V3Scale(V3Sub(maximum, minimum), half);
			const FloatV boxSphereR = V3Length(v);

			const Vec3V w = V3Sub(center, boxSphereCenter);
			const FloatV wLength = V3Length(w);
			const FloatV dif = FSub(FSub(radius, wLength), boxSphereR); 

			return FAllGrtrOrEq(dif, zero) != 0;
		}           
                  
		/**
		\brief Tests if the sphere intersects another sphere

		\param		sphere	[in] the other sphere
		\return		true if spheres overlap
		*/
		EHDI bool intersect(const SphereV& sphere) const
		{
			using namespace aos;
			const Vec3V centerDif = V3Sub(center, sphere.center);
			const FloatV cc = V3Dot(centerDif, centerDif);
			const FloatV r = FAdd(radius, sphere.radius);
			const FloatV rr = FMul(r, r);
			return FAllGrtrOrEq(rr, cc) != 0;
		}

		//return point in local space
		EHD aos::Vec3V getPoint(const PxU8)
		{
			return aos::V3Zero();
		}
  //  
		//sweep code need to have full version
		EHD aos::Vec3V supportSweep(const aos::Vec3VArg dir)const
		{
			using namespace aos;
			const Vec3V _dir = V3Normalize(dir);
			return V3ScaleAdd(_dir, radius, center);  
		}

		//make the support function the same as support margin
		EHD aos::Vec3V support(const aos::Vec3VArg)const
		{
			return center;//_margin is the same as radius
		}
  

		EHD aos::Vec3V supportMargin(const aos::Vec3VArg dir, const aos::FloatVArg _margin, aos::Vec3V& support)const
		{
			E_UNUSED(_margin);
			E_UNUSED(dir);

			support = center;
			return center;//_margin is the same as radius
		}

		EHD aos::BoolV isMarginEqRadius()const
		{
			return aos::BTTTT();
		}

		EHD aos::FloatV getSweepMargin() const
		{
			return aos::FZero();
		}


		aos::FloatV radius;		//!< Sphere's center, w component is radius

	};
}

}

// ===== 원본: physx/source/geomutils/src/convex/GuConvexHelper.h
namespace px
{
	class PxConvexMeshGeometry;
namespace Gu
{
	///////////////////////////////////////////////////////////////////////////

	EHDV void getScaledConvex(	PxVec3*& scaledVertices, PxU8*& scaledIndices, PxVec3* dstVertices, PxU8* dstIndices,
							bool idtConvexScale, const PxVec3* srcVerts, const PxU8* srcIndices, PxU32 nbVerts, const Cm::FastVertex2ShapeScaling& convexScaling);

	// PT: calling this correctly isn't trivial so let's macroize it. At least we limit the damage since it immediately calls a real function.
	#define GET_SCALEX_CONVEX(scaledVertices, stackIndices, idtScaling, nbVerts, scaling, srcVerts, srcIndices)	\
	getScaledConvex(scaledVertices, stackIndices,																\
					idtScaling ? NULL : reinterpret_cast<PxVec3*>(PxAlloca(nbVerts * sizeof(PxVec3))),			\
					idtScaling ? NULL : reinterpret_cast<PxU8*>(PxAlloca(nbVerts * sizeof(PxU8))),				\
					idtScaling, srcVerts, srcIndices, nbVerts, scaling);

	EHDV bool getConvexData(const PxConvexMeshGeometry& shapeConvex, Cm::FastVertex2ShapeScaling& scaling, PxBounds3& bounds, PolygonalData& polyData);

	struct ConvexEdge
	{
		PxU8	vref0;
		PxU8	vref1;
		PxVec3	normal;	// warning: non-unit vector!
	};

	EHDV PxU32 findUniqueConvexEdges(PxU32 maxNbEdges, ConvexEdge*  edges, PxU32 numPolygons, const Gu::HullPolygonData*  polygons, const PxU8*  vertexData);
}
}

// ===== 원본: physx/source/geomutils/src/convex/GuConvexHelper.cpp
using namespace px;
using namespace Gu;

// PT: we can't call alloca in a function and we want to avoid defines or duplicating the code. This makes it a bit tricky to write.
EHDI void Gu::getScaledConvex(	PxVec3*& scaledVertices, PxU8*& scaledIndices, PxVec3* dstVertices, PxU8* dstIndices,
							bool idtConvexScale, const PxVec3* srcVerts, const PxU8* srcIndices, PxU32 nbVerts, const Cm::FastVertex2ShapeScaling& convexScaling)
{
	//pretransform convex polygon if we have scaling!
	if(idtConvexScale)	// PT: the scale is always 1 for boxes so no need to test the type
	{
		scaledVertices = const_cast<PxVec3*>(srcVerts);
		scaledIndices = const_cast<PxU8*>(srcIndices);
	}
	else
	{
		scaledIndices = dstIndices;
		scaledVertices = dstVertices;
		for(PxU32 i=0; i<nbVerts; i++)
		{
			scaledIndices[i] = PxTo8(i);	//generate trivial indexing.
			scaledVertices[i] = convexScaling * srcVerts[srcIndices[i]];
		}
	}
}

EHDI bool Gu::getConvexData(const PxConvexMeshGeometry& shapeConvex, Cm::FastVertex2ShapeScaling& scaling, PxBounds3& bounds, PolygonalData& polyData)
{
	const bool idtScale = shapeConvex.scale.isIdentity();
	if(!idtScale)
		scaling.init(shapeConvex.scale);

	// PT: this version removes all the FCMPs and almost all LHS. This is temporary until
	// the legacy 3x3 matrix totally vanishes but meanwhile do NOT do useless matrix conversions,
	// it's a perfect recipe for LHS.
	const ConvexHullData* hullData = _getHullData(shapeConvex);
	E_ASSERT(!hullData->mAABB.isEmpty());
	bounds = hullData->mAABB.transformFast(scaling.getVertex2ShapeSkew());

	getPolygonalData_Convex(&polyData, hullData, scaling);

	// PT: non-uniform scaling invalidates the "internal objects" optimization, since our internal sphere
	// might become an ellipsoid or something. Just disable the optimization if scaling is used...
	if(!idtScale)
		polyData.mInternal.reset();

	return idtScale;
}

EHDI PxU32 Gu::findUniqueConvexEdges(PxU32 maxNbEdges, ConvexEdge*  edges, PxU32 numPolygons, const Gu::HullPolygonData*  polygons, const PxU8*  vertexData)
{
	PxU32 nbEdges = 0;

	while(numPolygons--)
	{
		const HullPolygonData& polygon = *polygons++;
		const PxU8* vRefBase = vertexData + polygon.mVRef8;
		PxU32 numEdges = polygon.mNbVerts;

		PxU32 a = numEdges - 1;
		PxU32 b = 0;
		while(numEdges--)
		{
			PxU8 vi0 =  vRefBase[a];
			PxU8 vi1 =	vRefBase[b];

			if(vi1 < vi0)
			{
				PxU8 tmp = vi0;
				vi0 = vi1;
				vi1 = tmp;
			}

			bool found=false;
			for(PxU32 i=0;i<nbEdges;i++)
			{
				if(edges[i].vref0==vi0 && edges[i].vref1==vi1)
				{
					found = true;
					edges[i].normal += polygon.mPlane.n;
					break;
				}
			}
			if(!found)
			{
				if(nbEdges==maxNbEdges)
				{
					EPX_ALWAYS_ASSERT_MESSAGE("Internal error: max nb edges reached. This shouldn't be possible...");
					return nbEdges;
				}

				edges[nbEdges].vref0	= vi0;
				edges[nbEdges].vref1	= vi1;
				edges[nbEdges].normal	= polygon.mPlane.n;
				nbEdges++;
			}

			a = b;
			b++;
		}
	}
	return nbEdges;
}

// ===== 원본: physx/source/geomutils/include/GuSegment.h
namespace px
{
namespace Gu
{

	/**
	\brief Represents a line segment.

	Line segment geometry
	In some cases this structure will be used to represent the infinite line that passes point0 and point1.
	*/
	class Segment
	{
	public:
		/**
		\brief Constructor
		*/
		EHDI Segment()
		{
		}

		/**
		\brief Constructor
		*/
		EHDI Segment(const PxVec3& _p0, const PxVec3& _p1) : p0(_p0), p1(_p1)
		{
		}

		/**
		\brief Copy constructor
		*/
		EHDI Segment(const Segment& seg) : p0(seg.p0), p1(seg.p1)
		{
		}

		/**
		\brief Destructor
		*/
		EHDI ~Segment()
		{
		}

		//! Assignment operator
		EHDI Segment& operator=(const Segment& other)
		{
			p0 = other.p0; 
			p1 = other.p1; 
			return *this;
		}

		//! Equality operator
		EHDI bool operator==(const Segment& other) const
		{
			return (p0==other.p0 && p1==other.p1);
		}

		//! Inequality operator
		EHDI bool operator!=(const Segment& other) const
		{
			return (p0!=other.p0 || p1!=other.p1);
		}

		EHDI const PxVec3& getOrigin() const
		{
			return p0;
		}

		//! Return the vector from point0 to point1
		EHDI PxVec3 computeDirection() const
		{
			return p1 - p0;
		}

		//! Return the vector from point0 to point1
		EHDI void computeDirection(PxVec3& dir) const
		{
			dir = p1 - p0;
		}

		//! Return the center of the segment segment
		EHDI PxVec3 computeCenter() const
		{
			return (p0 + p1)*0.5f;
		}

		EHDI PxF32 computeLength() const
		{
			return (p1-p0).magnitude();
		}

		EHDI PxF32 computeSquareLength() const
		{
			return (p1-p0).magnitudeSquared();
		}

		// PT: TODO: remove this one
		//! Return the square of the length of vector from point0 to point1
		EHDI PxReal lengthSquared() const
		{
			return ((p1 - p0).magnitudeSquared());
		}

		// PT: TODO: remove this one
		//! Return the length of vector from point0 to point1
		EHDI PxReal length() const
		{
			return ((p1 - p0).magnitude());
		}

		/*		EHDI void setOriginDirection(const PxVec3& origin, const PxVec3& direction)
		{
		p0 = p1 = origin;
		p1 += direction;
		}*/

		/**
		\brief Computes a point on the segment

		\param[out] pt point on segment
		\param[in] t point's parameter [t=0 => pt = mP0, t=1 => pt = mP1]
		*/
		EHDI void computePoint(PxVec3& pt, PxF32 t) const
		{
			pt = p0 + t * (p1 - p0);
		}

		// PT: TODO: remove this one
		//! Return the point at parameter t along the line: point0 + t*(point1-point0)
		EHDI PxVec3 getPointAt(PxReal t) const
		{
			return (p1 - p0)*t + p0;
		}

		PxVec3	p0;		//!< Start of segment
		PxVec3	p1;		//!< End of segment
	};
	E_STATIC_ASSERT(sizeof(Gu::Segment) == 24);
}

}

// ===== 원본: physx/source/geomutils/include/GuDistanceSegmentSegment.h
namespace px
{
namespace Gu
{
	// This version fixes accuracy issues (e.g.  TTP 4617), but needs to do 2 square roots in order
	// to find the normalized direction and length of the segments, and then
	// a division in order to renormalize the output

		EHDV PxReal	distanceSegmentSegmentSquared(	const PxVec3& origin0, const PxVec3& dir0, PxReal extent0,
																const PxVec3& origin1, const PxVec3& dir1, PxReal extent1,
																PxReal* s=NULL, PxReal* t=NULL);

	 EHDV PxReal	distanceSegmentSegmentSquared(	const PxVec3& origin0, const PxVec3& extent0,
																const PxVec3& origin1, const PxVec3& extent1,
																PxReal* s=NULL, PxReal* t=NULL);

	EHD	PxReal	distanceSegmentSegmentSquared(	const Gu::Segment& segment0,
															const Gu::Segment& segment1,
															PxReal* s=NULL, PxReal* t=NULL)
	{
		return distanceSegmentSegmentSquared(	segment0.p0, segment0.computeDirection(),
												segment1.p0, segment1.computeDirection(),
												s, t);
	}

	 EHDV aos::FloatV distanceSegmentSegmentSquared(	const aos::Vec3VArg p1, const aos::Vec3VArg d1, const aos::Vec3VArg p2, const aos::Vec3VArg d2,
					 								aos::FloatV& param0, 
					 								aos::FloatV& param1);

	// This function do four segment segment closest point test in one go
	EHDV aos::Vec4V distanceSegmentSegmentSquared4(  const aos::Vec3VArg p, const aos::Vec3VArg d, 
												const aos::Vec3VArg p02, const aos::Vec3VArg d02, 
												const aos::Vec3VArg p12, const aos::Vec3VArg d12, 
												const aos::Vec3VArg p22, const aos::Vec3VArg d22,
												const aos::Vec3VArg p32, const aos::Vec3VArg d32,
												aos::Vec4V& s, aos::Vec4V& t);

} // namespace Gu
}

// ===== 원본: physx/source/geomutils/src/distance/GuDistanceSegmentSegment.cpp
using namespace px;
using namespace aos;

static const float ZERO_TOLERANCE = 1e-06f;

// S0 = origin + extent * dir;
// S1 = origin - extent * dir;
EHDI PxReal Gu::distanceSegmentSegmentSquared(	const PxVec3& origin0, const PxVec3& dir0, PxReal extent0,
											const PxVec3& origin1, const PxVec3& dir1, PxReal extent1,
											PxReal* param0, PxReal* param1)
{
    const PxVec3 kDiff	= origin0 - origin1;
    const PxReal fA01	= -dir0.dot(dir1);
    const PxReal fB0	= kDiff.dot(dir0);
    const PxReal fB1	= -kDiff.dot(dir1);
	const PxReal fC		= kDiff.magnitudeSquared();
	const PxReal fDet	= PxAbs(1.0f - fA01*fA01);
    PxReal fS0, fS1, fSqrDist, fExtDet0, fExtDet1, fTmpS0, fTmpS1;

    if (fDet >= ZERO_TOLERANCE)
    {
        // segments are not parallel
        fS0 = fA01*fB1-fB0;
        fS1 = fA01*fB0-fB1;
        fExtDet0 = extent0*fDet;
        fExtDet1 = extent1*fDet;

        if (fS0 >= -fExtDet0)
        {
            if (fS0 <= fExtDet0)
            {
                if (fS1 >= -fExtDet1)
                {
                    if (fS1 <= fExtDet1)  // region 0 (interior)
                    {
                        // minimum at two interior points of 3D lines
                        PxReal fInvDet = 1.0f/fDet;
                        fS0 *= fInvDet;
                        fS1 *= fInvDet;
                        fSqrDist = fS0*(fS0+fA01*fS1+2.0f*fB0) + fS1*(fA01*fS0+fS1+2.0f*fB1)+fC;
                    }
                    else  // region 3 (side)
                    {
                        fS1 = extent1;
                        fTmpS0 = -(fA01*fS1+fB0);
                        if (fTmpS0 < -extent0)
                        {
                            fS0 = -extent0;
                            fSqrDist = fS0*(fS0-2.0f*fTmpS0) + fS1*(fS1+2.0f*fB1)+fC;
                        }
                        else if (fTmpS0 <= extent0)
                        {
                            fS0 = fTmpS0;
                            fSqrDist = -fS0*fS0+fS1*(fS1+2.0f*fB1)+fC;
                        }
                        else
                        {
                            fS0 = extent0;
                            fSqrDist = fS0*(fS0-2.0f*fTmpS0) + fS1*(fS1+2.0f*fB1)+fC;
                        }
                    }
                }
                else  // region 7 (side)
                {
                    fS1 = -extent1;
                    fTmpS0 = -(fA01*fS1+fB0);
                    if (fTmpS0 < -extent0)
                    {
                        fS0 = -extent0;
                        fSqrDist = fS0*(fS0-2.0f*fTmpS0) + fS1*(fS1+2.0f*fB1)+fC;
                    }
                    else if (fTmpS0 <= extent0)
                    {
                        fS0 = fTmpS0;
                        fSqrDist = -fS0*fS0+fS1*(fS1+2.0f*fB1)+fC;
                    }
                    else
                    {
                        fS0 = extent0;
                        fSqrDist = fS0*(fS0-2.0f*fTmpS0) + fS1*(fS1+2.0f*fB1)+fC;
                    }
                }
            }
            else
            {
                if (fS1 >= -fExtDet1)
                {
                    if (fS1 <= fExtDet1)  // region 1 (side)
                    {
                        fS0 = extent0;
                        fTmpS1 = -(fA01*fS0+fB1);
                        if (fTmpS1 < -extent1)
                        {
                            fS1 = -extent1;
                            fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                        }
                        else if (fTmpS1 <= extent1)
                        {
                            fS1 = fTmpS1;
                            fSqrDist = -fS1*fS1+fS0*(fS0+2.0f*fB0)+fC;
                        }
                        else
                        {
                            fS1 = extent1;
                            fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                        }
                    }
                    else  // region 2 (corner)
                    {
                        fS1 = extent1;
                        fTmpS0 = -(fA01*fS1+fB0);
                        if (fTmpS0 < -extent0)
                        {
                            fS0 = -extent0;
                            fSqrDist = fS0*(fS0-2.0f*fTmpS0) + fS1*(fS1+2.0f*fB1)+fC;
                        }
                        else if (fTmpS0 <= extent0)
                        {
                            fS0 = fTmpS0;
                            fSqrDist = -fS0*fS0+fS1*(fS1+2.0f*fB1)+fC;
                        }
                        else
                        {
                            fS0 = extent0;
                            fTmpS1 = -(fA01*fS0+fB1);
                            if (fTmpS1 < -extent1)
                            {
                                fS1 = -extent1;
                                fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                            }
                            else if (fTmpS1 <= extent1)
                            {
                                fS1 = fTmpS1;
                                fSqrDist = -fS1*fS1+fS0*(fS0+2.0f*fB0) + fC;
                            }
                            else
                            {
                                fS1 = extent1;
                                fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                            }
                        }
                    }
                }
                else  // region 8 (corner)
                {
                    fS1 = -extent1;
                    fTmpS0 = -(fA01*fS1+fB0);
                    if (fTmpS0 < -extent0)
                    {
                        fS0 = -extent0;
                        fSqrDist = fS0*(fS0-2.0f*fTmpS0) + fS1*(fS1+2.0f*fB1)+fC;
                    }
                    else if (fTmpS0 <= extent0)
                    {
                        fS0 = fTmpS0;
                        fSqrDist = -fS0*fS0+fS1*(fS1+2.0f*fB1)+fC;
                    }
                    else
                    {
                        fS0 = extent0;
                        fTmpS1 = -(fA01*fS0+fB1);
                        if (fTmpS1 > extent1)
                        {
                            fS1 = extent1;
                            fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                        }
                        else if (fTmpS1 >= -extent1)
                        {
                            fS1 = fTmpS1;
                            fSqrDist = -fS1*fS1+fS0*(fS0+2.0f*fB0) + fC;
                        }
                        else
                        {
                            fS1 = -extent1;
                            fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                        }
                    }
                }
            }
        }
        else 
        {
            if (fS1 >= -fExtDet1)
            {
                if (fS1 <= fExtDet1)  // region 5 (side)
                {
                    fS0 = -extent0;
                    fTmpS1 = -(fA01*fS0+fB1);
                    if (fTmpS1 < -extent1)
                    {
                        fS1 = -extent1;
                        fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                    }
                    else if (fTmpS1 <= extent1)
                    {
                        fS1 = fTmpS1;
                        fSqrDist = -fS1*fS1+fS0*(fS0+2.0f*fB0)+fC;
                    }
                    else
                    {
                        fS1 = extent1;
                        fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                    }
                }
                else  // region 4 (corner)
                {
                    fS1 = extent1;
                    fTmpS0 = -(fA01*fS1+fB0);
                    if (fTmpS0 > extent0)
                    {
                        fS0 = extent0;
                        fSqrDist = fS0*(fS0-2.0f*fTmpS0) + fS1*(fS1+2.0f*fB1)+fC;
                    }
                    else if (fTmpS0 >= -extent0)
                    {
                        fS0 = fTmpS0;
                        fSqrDist = -fS0*fS0+fS1*(fS1+2.0f*fB1)+fC;
                    }
                    else
                    {
                        fS0 = -extent0;
                        fTmpS1 = -(fA01*fS0+fB1);
                        if (fTmpS1 < -extent1)
                        {
                            fS1 = -extent1;
                            fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                        }
                        else if (fTmpS1 <= extent1)
                        {
                            fS1 = fTmpS1;
                            fSqrDist = -fS1*fS1+fS0*(fS0+2.0f*fB0) + fC;
                        }
                        else
                        {
                            fS1 = extent1;
                            fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                        }
                    }
                }
            }
            else   // region 6 (corner)
            {
                fS1 = -extent1;
                fTmpS0 = -(fA01*fS1+fB0);
                if (fTmpS0 > extent0)
                {
                    fS0 = extent0;
                    fSqrDist = fS0*(fS0-2.0f*fTmpS0) + fS1*(fS1+2.0f*fB1)+fC;
                }
                else if (fTmpS0 >= -extent0)
                {
                    fS0 = fTmpS0;
                    fSqrDist = -fS0*fS0+fS1*(fS1+2.0f*fB1)+fC;
                }
                else
                {
                    fS0 = -extent0;
                    fTmpS1 = -(fA01*fS0+fB1);
                    if (fTmpS1 < -extent1)
                    {
                        fS1 = -extent1;
                        fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                    }
                    else if (fTmpS1 <= extent1)
                    {
                        fS1 = fTmpS1;
                        fSqrDist = -fS1*fS1+fS0*(fS0+2.0f*fB0) + fC;
                    }
                    else
                    {
                        fS1 = extent1;
                        fSqrDist = fS1*(fS1-2.0f*fTmpS1) + fS0*(fS0+2.0f*fB0)+fC;
                    }
                }
            }
        }
    }
    else
    {
		// The segments are parallel.
		PxReal fE0pE1 = extent0 + extent1;
		PxReal fSign = (fA01 > 0.0f ? -1.0f : 1.0f);
		PxReal b0Avr = 0.5f*(fB0 - fSign*fB1);
		PxReal fLambda = -b0Avr;
		if(fLambda < -fE0pE1)
		{
			fLambda = -fE0pE1;
		}
		else if(fLambda > fE0pE1)
		{
			fLambda = fE0pE1;
		}

		fS1 = -fSign*fLambda*extent1/fE0pE1;
		fS0 = fLambda + fSign*fS1;
		fSqrDist = fLambda*(fLambda + 2.0f*b0Avr) + fC;
	}

	if(param0)
		*param0 = fS0;
	if(param1)
		*param1 = fS1;

	// account for numerical round-off error
	return px::intrinsics::selectMax(0.0f, fSqrDist);
}

EHDI PxReal Gu::distanceSegmentSegmentSquared(	const PxVec3& origin0, const PxVec3& extent0,
											const PxVec3& origin1, const PxVec3& extent1,
											PxReal* param0, 
											PxReal* param1)
{
	// Some conversion is needed between the old & new code
	// Old:
	// segment (s0, s1)
	// origin = s0
	// extent = s1 - s0
	//
	// New:
	// s0 = origin + extent * dir;
	// s1 = origin - extent * dir;

	// dsequeira: is this really sensible? We use a highly optimized Wild Magic routine, 
	// then use a segment representation that requires an expensive conversion to/from...

	PxVec3 dir0 = extent0;
	const PxVec3 center0 = origin0 + extent0*0.5f;
	PxReal length0 = extent0.magnitude();	//AM: change to make it work for degenerate (zero length) segments.
	const bool b0 = length0 != 0.0f;
	PxReal oneOverLength0 = 0.0f;
	if(b0)
	{
		oneOverLength0 = 1.0f / length0;
		dir0 *= oneOverLength0;
		length0 *= 0.5f;
	}

	PxVec3 dir1 = extent1;
	const PxVec3 center1 = origin1 + extent1*0.5f;
	PxReal length1 = extent1.magnitude();
	const bool b1 = length1 != 0.0f;
	PxReal oneOverLength1 = 0.0f;
	if(b1)
	{
		oneOverLength1 = 1.0f / length1;
		dir1 *= oneOverLength1;
		length1 *= 0.5f;
	}

	// the return param vals have -extent = s0, extent = s1

	const PxReal d2 = distanceSegmentSegmentSquared(center0, dir0, length0,
													center1, dir1, length1,
													param0, param1);

	//ML : This is wrong for some reason, I guess it has precision issue
	//// renormalize into the 0 = s0, 1 = s1 range
	//if (param0)
	//	*param0 = b0 ? ((*param0) * oneOverLength0 * 0.5f + 0.5f) : 0.0f;
	//if (param1)
	//	*param1 = b1 ? ((*param1) * oneOverLength1 * 0.5f + 0.5f) : 0.0f;

	if(param0)
		*param0 = b0 ? ((length0 + (*param0))*oneOverLength0) : 0.0f;
	if(param1)
		*param1 = b1 ? ((length1 + (*param1))*oneOverLength1) : 0.0f;

	return d2;
}

/*
	S0 = origin + extent * dir;
	S1 = origin + extent * dir;
	dir is the vector from start to end point
	p1 is the start point of segment1
	d1 is the direction vector(q1 - p1)
	p2 is the start point of segment2
	d2 is the direction vector(q2 - p2) 
*/

EHDI FloatV Gu::distanceSegmentSegmentSquared(	const Vec3VArg p1, 
											const Vec3VArg d1,
											const Vec3VArg p2, 
											const Vec3VArg d2,
											FloatV& s, 
											FloatV& t)
{
	const FloatV zero = FZero();
	const FloatV one = FOne();
	const FloatV eps = FEps();

	const Vec3V r = V3Sub(p1, p2);
	const Vec4V combinedDot = V3Dot4(d1, d1, d2, d2, d1, d2, d1, r);
	const Vec4V combinedRecip = V4Sel(V4IsGrtr(combinedDot, V4Splat(eps)), V4Recip(combinedDot), V4Splat(zero));
	const FloatV a = V4GetX(combinedDot);
	const FloatV e = V4GetY(combinedDot);
	const FloatV b = V4GetZ(combinedDot);
	const FloatV c = V4GetW(combinedDot);
	const FloatV aRecip = V4GetX(combinedRecip);//FSel(FIsGrtr(a, eps), FRecip(a), zero);
	const FloatV eRecip = V4GetY(combinedRecip);//FSel(FIsGrtr(e, eps), FRecip(e), zero);

	const FloatV f = V3Dot(d2, r);

	/*
		s = (b*f - c*e)/(a*e - b*b);
		t = (a*f - b*c)/(a*e - b*b);

		s = (b*t - c)/a;
		t = (b*s + f)/e;
	*/
	
	//if segments not parallel, the general non-degenerated case, compute closest point on two segments and clamp to segment1
	const FloatV denom = FSub(FMul(a, e), FMul(b, b));
	const FloatV temp = FSub(FMul(b, f), FMul(c, e));
	const FloatV s0 = FClamp(FDiv(temp, denom), zero, one);
	
	//if segment is parallel, demon < eps
	const BoolV con2 = FIsGrtr(eps, denom);//FIsEq(denom, zero);
	const FloatV sTmp = FSel(con2, FHalf(), s0);
	
	//compute point on segment2 closest to segment1
	//const FloatV tTmp = FMul(FAdd(FMul(b, sTmp), f), eRecip);
	const FloatV tTmp = FMul(FScaleAdd(b, sTmp, f), eRecip);

	//if t is in [zero, one], done. otherwise clamp t
	const FloatV t2 = FClamp(tTmp, zero, one);

	//recompute s for the new value
	const FloatV comp = FMul(FSub(FMul(b,t2), c), aRecip);
	const FloatV s2 = FClamp(comp, zero, one);

	s = s2;
	t = t2;

	const Vec3V closest1 = V3ScaleAdd(d1, s2, p1);//V3Add(p1, V3Scale(d1, tempS));
	const Vec3V closest2 = V3ScaleAdd(d2, t2, p2);//V3Add(p2, V3Scale(d2, tempT));
	const Vec3V vv = V3Sub(closest1, closest2);
	return V3Dot(vv, vv);
}




/*
	segment (p, d) and segment (p02, d02)
	segment (p, d) and segment (p12, d12)
	segment (p, d) and segment (p22, d22)
	segment (p, d) and segment (p32, d32)
*/
EHDI Vec4V Gu::distanceSegmentSegmentSquared4(	const Vec3VArg p, const Vec3VArg d0, 
											const Vec3VArg p02, const Vec3VArg d02, 
											const Vec3VArg p12, const Vec3VArg d12, 
											const Vec3VArg p22, const Vec3VArg d22,
											const Vec3VArg p32, const Vec3VArg d32,
											Vec4V& s, Vec4V& t)
{
	const Vec4V zero = V4Zero();
	const Vec4V one = V4One();
	const Vec4V eps = V4Eps();
	const Vec4V half = V4Splat(FHalf());

	const Vec4V d0X = V4Splat(V3GetX(d0));
	const Vec4V d0Y = V4Splat(V3GetY(d0));
	const Vec4V d0Z = V4Splat(V3GetZ(d0));
	const Vec4V pX  = V4Splat(V3GetX(p));
	const Vec4V pY  = V4Splat(V3GetY(p));
	const Vec4V pZ  = V4Splat(V3GetZ(p));

	Vec4V d024 = Vec4V_From_Vec3V(d02);
	Vec4V d124 = Vec4V_From_Vec3V(d12);
	Vec4V d224 = Vec4V_From_Vec3V(d22);
	Vec4V d324 = Vec4V_From_Vec3V(d32);

	Vec4V p024 = Vec4V_From_Vec3V(p02);
	Vec4V p124 = Vec4V_From_Vec3V(p12);
	Vec4V p224 = Vec4V_From_Vec3V(p22);
	Vec4V p324 = Vec4V_From_Vec3V(p32);

	Vec4V d0123X, d0123Y, d0123Z;
	Vec4V p0123X, p0123Y, p0123Z;

	E_TRANSPOSE_44_34(d024, d124, d224, d324, d0123X, d0123Y, d0123Z);
	E_TRANSPOSE_44_34(p024, p124, p224, p324, p0123X, p0123Y, p0123Z);

	const Vec4V rX = V4Sub(pX, p0123X);
	const Vec4V rY = V4Sub(pY, p0123Y);
	const Vec4V rZ = V4Sub(pZ, p0123Z);

	//TODO - store this in a transposed state and avoid so many dot products?

	const FloatV dd = V3Dot(d0, d0);

	const Vec4V e = V4MulAdd(d0123Z, d0123Z, V4MulAdd(d0123X, d0123X, V4Mul(d0123Y, d0123Y)));
	const Vec4V b = V4MulAdd(d0Z, d0123Z, V4MulAdd(d0X, d0123X, V4Mul(d0Y, d0123Y)));
	const Vec4V c = V4MulAdd(d0Z, rZ, V4MulAdd(d0X, rX, V4Mul(d0Y, rY)));
	const Vec4V f = V4MulAdd(d0123Z, rZ, V4MulAdd(d0123X, rX, V4Mul(d0123Y, rY))); 

	const Vec4V a(V4Splat(dd));

	const Vec4V aRecip(V4Recip(a));
	const Vec4V eRecip(V4Recip(e));

	//if segments not parallell, compute closest point on two segments and clamp to segment1
	const Vec4V denom = V4Sub(V4Mul(a, e), V4Mul(b, b));
	const Vec4V temp = V4Sub(V4Mul(b, f), V4Mul(c, e));
	const Vec4V s0 = V4Clamp(V4Div(temp, denom), zero, one);

	//test whether segments are parallel
	const BoolV con2 = V4IsGrtrOrEq(eps, denom);     
	const Vec4V sTmp = V4Sel(con2, half, s0);

	//compute point on segment2 closest to segment1
	const Vec4V tTmp = V4Mul(V4Add(V4Mul(b, sTmp), f), eRecip);

	//if t is in [zero, one], done. otherwise clamp t
	const Vec4V t2 = V4Clamp(tTmp, zero, one);

	//recompute s for the new value
	const Vec4V comp = V4Mul(V4Sub(V4Mul(b,t2), c), aRecip);
	const BoolV aaNearZero = V4IsGrtrOrEq(eps, a); // check if aRecip is valid (aa>eps)
	const Vec4V s2 = V4Sel(aaNearZero, V4Zero(), V4Clamp(comp, zero, one));

	/*  s = V4Sel(con0, zero, V4Sel(con1, cd, s2));
	t = V4Sel(con1, zero, V4Sel(con0, cg, t2));  */
	s = s2;
	t = t2;

	const Vec4V closest1X = V4MulAdd(d0X, s2, pX);
	const Vec4V closest1Y = V4MulAdd(d0Y, s2, pY);
	const Vec4V closest1Z = V4MulAdd(d0Z, s2, pZ);

	const Vec4V closest2X = V4MulAdd(d0123X, t2, p0123X);
	const Vec4V closest2Y = V4MulAdd(d0123Y, t2, p0123Y);
	const Vec4V closest2Z = V4MulAdd(d0123Z, t2, p0123Z);

	const Vec4V vvX = V4Sub(closest1X, closest2X);
	const Vec4V vvY = V4Sub(closest1Y, closest2Y);
	const Vec4V vvZ = V4Sub(closest1Z, closest2Z);

	const Vec4V vd = V4MulAdd(vvX, vvX, V4MulAdd(vvY, vvY, V4Mul(vvZ, vvZ)));

	return vd;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactGenSphereCapsule.cpp
using namespace px;
using namespace Gu;
using namespace aos;

//ML: this function is shared by the sphere/capsule vs convex hulls full contact gen, capsule in the local space of polyData
EHDI static bool testPolyDataAxis(const CapsuleV& capsule, const PolygonalData& polyData, const SupportLocal* map, const FloatVArg contactDist, FloatV& minOverlap, Vec3V& separatingAxis)
{
	FloatV _minOverlap = FMax();//minOverlap;
	FloatV min0, max0;
	FloatV min1, max1;
	Vec3V tempAxis = V3UnitY();

	//capsule in the local space of polyData
	for(PxU32 i=0; i<polyData.mNbPolygons; ++i)
	{
		const HullPolygonData& polygon = polyData.mPolygons[i];

		const Vec3V minVert = V3LoadU_SafeReadW(polyData.mVerts[polygon.mMinIndex]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData
		const FloatV planeDist = FLoad(polygon.mPlane.d);
		const Vec3V vertexSpacePlaneNormal = V3LoadU_SafeReadW(polygon.mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class

		//transform plane n to shape space
		const Vec3V shapeSpacePlaneNormal = M33TrnspsMulV3(map->shape2Vertex, vertexSpacePlaneNormal);

		const FloatV magnitude = FRecip(V3Length(shapeSpacePlaneNormal));
		//normalize shape space normal
		const Vec3V planeN = V3Scale(shapeSpacePlaneNormal, magnitude);
		//ML::use this to avoid LHS
		min0 = FMul(V3Dot(vertexSpacePlaneNormal, minVert), magnitude);
		max0 = FMul(FNeg(planeDist), magnitude);

		const FloatV tempMin = V3Dot(capsule.p0, planeN);
		const FloatV tempMax = V3Dot(capsule.p1, planeN);
		min1 = FMin(tempMin, tempMax);
		max1 = FMax(tempMin, tempMax);

		min1 = FSub(min1, capsule.radius);
		max1 = FAdd(max1, capsule.radius);

		const BoolV con = BOr(FIsGrtr(min1, FAdd(max0, contactDist)), FIsGrtr(min0, FAdd(max1, contactDist)));

		if(BAllEqTTTT(con))
			return false;

		const FloatV tempOverlap = FSub(max0, min1);

		if(FAllGrtr(_minOverlap, tempOverlap))
		{
			_minOverlap = tempOverlap;
			tempAxis = planeN;
		}
	}

	separatingAxis = tempAxis;
	minOverlap = _minOverlap;

	return true;
}  

//ML: this function is shared by sphere/capsule. Point a and direction d need to be in the local space of polyData
EHDI static bool intersectRayPolyhedron(const Vec3VArg a, const Vec3VArg dir, const PolygonalData& polyData, FloatV& tEnter, FloatV& tExit)
{
	const FloatV zero = FZero();
	const FloatV eps = FLoad(1e-7f);
	FloatV tFirst = zero;
	FloatV tLast = FMax();

	for(PxU32 k=0; k<polyData.mNbPolygons; ++k)
	{
		const HullPolygonData& data = polyData.mPolygons[k];
		const Vec3V n = V3LoadU_SafeReadW(data.mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class

		FloatV d = FLoad(data.mPlane.d);
	
		const FloatV denominator = V3Dot(n, dir);
		const FloatV distToPlane = FAdd(V3Dot(n, a), d);
		
		if(FAllGrtr(eps, FAbs(denominator)))
		{
			if(FAllGrtr(distToPlane, zero))
				return false;
		}
		else
		{
			FloatV tTemp = FNeg(FDiv(distToPlane, denominator));

			//ML: denominator < 0 means the ray is entering halfspace; denominator > 0 means the ray is exiting halfspace
			const BoolV con = FIsGrtr(zero, denominator);
			const BoolV con0 = FIsGrtr(tTemp, tFirst); 
			const BoolV con1 = FIsGrtr(tLast, tTemp);
	
			tFirst = FSel(BAnd(con, con0), tTemp, tFirst);
			tLast = FSel(BAndNot(con1, con), tTemp, tLast);
		}
			
		if(FAllGrtr(tFirst, tLast))
			return false;
	}

	//calculate the intersect p in the local space
	tEnter = tFirst;
	tExit = tLast;
	
	return true;
}

//Precompute the convex data
//     7+------+6			0 = ---
//     /|     /|			1 = +--
//    / |    / |			2 = ++-
//   / 4+---/--+5			3 = -+-
// 3+------+2 /    y   z	4 = --+
//  | /    | /     |  /		5 = +-+
//  |/     |/      |/		6 = +++
// 0+------+1      *---x	7 = -++

EHDI static bool testSATCapsulePoly(const CapsuleV& capsule, const PolygonalData& polyData, const SupportLocal* map, const FloatVArg contactDist, FloatV& minOverlap, Vec3V& separatingAxis)
{
	FloatV _minOverlap = FMax();//minOverlap;
	FloatV min0, max0;
	FloatV min1, max1;
	Vec3V tempAxis = V3UnitY();
	const FloatV eps = FEps();

	//ML: project capsule to polydata axis 
	if(!testPolyDataAxis(capsule, polyData, map, contactDist, _minOverlap, tempAxis))
		return false;

	const Vec3V capsuleAxis = V3Sub(capsule.p1, capsule.p0);
	
	for(PxU32 i=0;i<polyData.mNbPolygons;i++)
	{
		const HullPolygonData& polygon = polyData.mPolygons[i];
		const PxU8* inds1 = polyData.mPolygonVertexRefs + polygon.mVRef8;

		for(PxU32 lStart = 0, lEnd = PxU32(polygon.mNbVerts-1); lStart<polygon.mNbVerts; lEnd = lStart++)
		{
			//in the vertex space
			const Vec3V p10 = V3LoadU_SafeReadW(polyData.mVerts[inds1[lStart]]);	// PT: safe because of the way vertex memory is allocated in ConvexHullData
			const Vec3V p11 = V3LoadU_SafeReadW(polyData.mVerts[inds1[lEnd]]);		// PT: safe because of the way vertex memory is allocated in ConvexHullData
				
			const Vec3V vertexSpaceV = V3Sub(p11, p10);

			//transform v to shape space
			const Vec3V shapeSpaceV = M33TrnspsMulV3(map->shape2Vertex, vertexSpaceV);
			
			const Vec3V dir = V3Cross(capsuleAxis, shapeSpaceV);
			const FloatV lenSq = V3Dot(dir, dir);
			if(FAllGrtr(eps, lenSq))
				continue;
			const Vec3V normal = V3ScaleInv(dir, FSqrt(lenSq));

			map->doSupport(normal, min0, max0);

			const FloatV tempMin = V3Dot(capsule.p0, normal);
			const FloatV tempMax = V3Dot(capsule.p1, normal);
			min1 = FMin(tempMin, tempMax);
			max1 = FMax(tempMin, tempMax);

			min1 = FSub(min1, capsule.radius);
			max1 = FAdd(max1, capsule.radius);

			const BoolV con = BOr(FIsGrtr(min1, FAdd(max0, contactDist)), FIsGrtr(min0, FAdd(max1, contactDist)));

			if(BAllEqTTTT(con))
				return false;
	
			const FloatV tempOverlap = FSub(max0, min1);

			if(FAllGrtr(_minOverlap, tempOverlap))
			{
				_minOverlap = tempOverlap;
				tempAxis = normal;
			}
		}
	}
		
	separatingAxis = tempAxis;
	minOverlap = _minOverlap;

	return true;
}

EHDI static void generatedCapsuleBoxFaceContacts(const CapsuleV& capsule, const PolygonalData& polyData, const HullPolygonData& referencePolygon, const SupportLocal* map, const PxMatTransformV& aToB, PersistentContact* manifoldContacts, PxU32& numContacts, 
	const FloatVArg contactDist, const Vec3VArg normal)
{
	const FloatV radius = FAdd(capsule.radius, contactDist);

	//calculate the intersect point of ray to plane
	const Vec3V planeNormal = V3Normalize(M33TrnspsMulV3(map->shape2Vertex, V3LoadU(referencePolygon.mPlane.n)));
	const PxU8* inds = polyData.mPolygonVertexRefs + referencePolygon.mVRef8;
	const Vec3V a = M33MulV3(map->vertex2Shape, V3LoadU_SafeReadW(polyData.mVerts[inds[0]]));	// PT: safe because of the way vertex memory is allocated in ConvexHullData
	
	const FloatV denom0 = V3Dot(planeNormal, V3Sub(capsule.p0, a)); 
	const FloatV denom1 = V3Dot(planeNormal, V3Sub(capsule.p1, a)); 
	const FloatV projPlaneN = V3Dot(planeNormal, normal);
	const FloatV numer = FSel(FIsGrtr(projPlaneN, FZero()), FRecip(projPlaneN), FZero());
	const FloatV t0 = FMul(denom0, numer);
	const FloatV t1 = FMul(denom1, numer);

	const BoolV con0 = FIsGrtrOrEq(radius, t0);
	const BoolV con1 = FIsGrtrOrEq(radius, t1);
	if(BAllEqTTTT(BOr(con0, con1)))
	{
		const Mat33V rot = findRotationMatrixFromZAxis(planeNormal);
		Vec3V points0In0_buf[E_ALLOCA_N]; Vec3V* points0In0 = points0In0_buf;
		map->populateVerts(inds, referencePolygon.mNbVerts, polyData.mVerts, points0In0);

		Vec3V rPolygonMin = V3Splat(FMax());
		Vec3V rPolygonMax = V3Neg(rPolygonMin);
		for(PxU32 i=0; i<referencePolygon.mNbVerts; ++i)
		{
			points0In0[i] = M33MulV3(rot, points0In0[i]);
			rPolygonMin = V3Min(rPolygonMin, points0In0[i]);
			rPolygonMax = V3Max(rPolygonMax, points0In0[i]);
		}

		if(BAllEqTTTT(con0))
		{
			//add contact
			const Vec3V proj = V3NegScaleSub(normal, t0, capsule.p0);//V3ScaleAdd(t0, normal, capsule.p0);
			//transform proj0 to 2D
			const Vec3V point = M33MulV3(rot, proj);

			if(contains(points0In0, referencePolygon.mNbVerts, point, rPolygonMin, rPolygonMax))
			{
				manifoldContacts[numContacts].mLocalPointA = aToB.transformInv(capsule.p0);
				manifoldContacts[numContacts].mLocalPointB = proj;
				manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(normal), t0);
			}
		}

		if(BAllEqTTTT(con1))
		{
			const Vec3V proj = V3NegScaleSub(normal, t1, capsule.p1);//V3ScaleAdd(t1, normal, capsule.p1);
			//transform proj0 to 2D
			const Vec3V point = M33MulV3(rot, proj);

			if(contains(points0In0, referencePolygon.mNbVerts, point, rPolygonMin, rPolygonMax))
			{
				manifoldContacts[numContacts].mLocalPointA = aToB.transformInv(capsule.p1);
				manifoldContacts[numContacts].mLocalPointB = proj;
				manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(normal),t1);
			}
		}
	}
}

EHDI static void generatedFaceContacts(const CapsuleV& capsule, const PolygonalData& polyData, const SupportLocal* map, const PxMatTransformV& aToB, PersistentContact* manifoldContacts, PxU32& numContacts, 
	const FloatVArg contactDist, const Vec3VArg normal)
{
	const FloatV zero = FZero();
	FloatV tEnter = zero, tExit = zero;
	const FloatV inflatedRadius = FAdd(capsule.radius, contactDist);
	const Vec3V dir = V3Neg(normal);
	const Vec3V vertexSpaceDir = M33MulV3(map->shape2Vertex, dir);
	const Vec3V p0 = M33MulV3(map->shape2Vertex, capsule.p0);

	if(intersectRayPolyhedron(p0, vertexSpaceDir, polyData, tEnter, tExit) && FAllGrtrOrEq(inflatedRadius, tEnter))
	{
		manifoldContacts[numContacts].mLocalPointA = aToB.transformInv(capsule.p0);
		manifoldContacts[numContacts].mLocalPointB = V3ScaleAdd(dir, tEnter, capsule.p0);
		manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(normal), tEnter);
	}

	const Vec3V p1 = M33MulV3(map->shape2Vertex, capsule.p1);
	if(intersectRayPolyhedron(p1, vertexSpaceDir, polyData, tEnter, tExit) && FAllGrtrOrEq(inflatedRadius, tEnter))
	{
		manifoldContacts[numContacts].mLocalPointA = aToB.transformInv(capsule.p1);
		manifoldContacts[numContacts].mLocalPointB = V3ScaleAdd(dir, tEnter, capsule.p1);
		manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(normal), tEnter);
	}	
}

EHDI static void generateEE(const Vec3VArg p, const Vec3VArg q, const Vec3VArg normal, const Vec3VArg a, const Vec3VArg b, const PxMatTransformV& aToB, 
	PersistentContact* manifoldContacts, PxU32& numContacts, const FloatVArg inflatedRadius)
{
	const FloatV zero = FZero();
	const FloatV expandedRatio = FLoad(0.005f);
	const Vec3V ab = V3Sub(b, a);
	const Vec3V n = V3Cross(ab, normal);
	const FloatV d = V3Dot(n, a);
	const FloatV np = V3Dot(n, p);
	const FloatV nq = V3Dot(n,q);
	const FloatV signP = FSub(np, d);
	const FloatV signQ = FSub(nq, d);
	const FloatV temp = FMul(signP, signQ);
	if(FAllGrtr(temp, zero)) return;//If both points in the same side as the plane, no intersect points
		
	// if colliding edge (p3,p4) and plane are parallel return no collision
	const Vec3V pq = V3Sub(q, p);
	const FloatV npq = V3Dot(n, pq); 
	if(FAllEq(npq, zero))	return;

	//calculate the intersect point in the segment pq with plane n(x - a).
	const FloatV segTValue = FDiv(FSub(d, np), npq);
	const Vec3V localPointA = V3ScaleAdd(pq, segTValue, p);

	//ML: ab, localPointA and normal is in the same plane, so that we can do 2D segment segment intersection
	//calculate a normal perpendicular to ray localPointA + normal*t, then do 2D segment segment intersection
	const Vec3V perNormal = V3Cross(normal, pq);
	const Vec3V ap = V3Sub(localPointA, a);
	const FloatV nom = V3Dot(perNormal, ap);
	const FloatV denom = V3Dot(perNormal, ab);

	const FloatV tValue = FDiv(nom, denom);

	const FloatV max = FAdd(FOne(), expandedRatio);
	const FloatV min = FSub(zero, expandedRatio);
	if(FAllGrtr(tValue, max) || FAllGrtr(min, tValue))
		return;

	const Vec3V v = V3NegScaleSub(ab, tValue, ap);
	const FloatV signedDist = V3Dot(v, normal);
	if(FAllGrtrOrEq(inflatedRadius, signedDist))
	{
		const Vec3V localPointB = V3Sub(localPointA, v);
		manifoldContacts[numContacts].mLocalPointA = aToB.transformInv(localPointA);
		manifoldContacts[numContacts].mLocalPointB = localPointB;
		manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(normal), signedDist);	
	}
}

EHDI static void generatedContactsEEContacts(const CapsuleV& capsule, const PolygonalData& polyData, const HullPolygonData& referencePolygon, const SupportLocal* map, const PxMatTransformV& aToB, 
	PersistentContact* manifoldContacts, PxU32& numContacts, const FloatVArg contactDist, const Vec3VArg contactNormal)
{
	const PxU8* inds = polyData.mPolygonVertexRefs + referencePolygon.mVRef8;

	Vec3V points0In0_buf[E_ALLOCA_N]; Vec3V* points0In0 = points0In0_buf;

	//Transform all the verts from vertex space to shape space
	map->populateVerts(inds, referencePolygon.mNbVerts, polyData.mVerts, points0In0);

	const FloatV inflatedRadius = FAdd(capsule.radius, contactDist);

	for (PxU32 rStart = 0, rEnd = PxU32(referencePolygon.mNbVerts - 1); rStart < referencePolygon.mNbVerts; rEnd = rStart++) 
	{
		generateEE(capsule.p0, capsule.p1, contactNormal,points0In0[rStart], points0In0[rEnd], aToB, manifoldContacts, numContacts, inflatedRadius);
	}
}

EHDI bool Gu::generateCapsuleBoxFullContactManifold(const CapsuleV& capsule, const PolygonalData& polyData, const SupportLocal* map, const PxMatTransformV& aToB, PersistentContact* manifoldContacts, PxU32& numContacts,
	const FloatVArg contactDist, Vec3V& normal, const Vec3VArg closest, PxReal margin, bool doOverlapTest, PxReal toleranceScale, PxRenderOutput* renderOutput)
{
	E_UNUSED(renderOutput);
	const PxU32 originalContacts = numContacts;

	const HullPolygonData* referencePolygon = NULL;

	if(doOverlapTest)
	{
		FloatV minOverlap;
		//overwrite the normal
		if(!testSATCapsulePoly(capsule, polyData, map, contactDist, minOverlap, normal))
			return false;

		referencePolygon = &polyData.mPolygons[getPolygonIndex(polyData, map, V3Neg(normal))];
	}
	else
	{
		const PxReal lowerEps = toleranceScale * PCM_WITNESS_POINT_LOWER_EPS;
		const PxReal upperEps = toleranceScale * PCM_WITNESS_POINT_UPPER_EPS;
		const PxReal tolerance = PxClamp(margin, lowerEps, upperEps);

		const PxU32 featureIndex = getWitnessPolygonIndex(polyData, map, V3Neg(normal), closest, tolerance);
		referencePolygon = &polyData.mPolygons[featureIndex];	
	}

	generatedCapsuleBoxFaceContacts(capsule, polyData, *referencePolygon, map, aToB, manifoldContacts, numContacts, contactDist, normal);
	const PxU32 faceContacts = numContacts - originalContacts;
	if(faceContacts < 2)
	{
		generatedContactsEEContacts(capsule, polyData, *referencePolygon, map, aToB, manifoldContacts, numContacts, contactDist, normal);
	}

	return true;
}

//capsule is in the local space of polyData
EHDI bool Gu::generateFullContactManifold(const CapsuleV& capsule, const PolygonalData& polyData, const SupportLocal* map, const PxMatTransformV& aToB, PersistentContact* manifoldContacts, PxU32& numContacts,
	const FloatVArg contactDist, Vec3V& normal, const Vec3VArg closest, PxReal margin, bool doOverlapTest, PxReal toleranceLength)
{
	const PxU32 originalContacts = numContacts;

	Vec3V tNormal = normal;
	
	if(doOverlapTest)
	{	
		FloatV minOverlap;
			
		//overwrite the normal with the sperating axis
		if(!testSATCapsulePoly(capsule, polyData, map, contactDist, minOverlap, tNormal))
			return false;

		generatedFaceContacts(capsule, polyData, map, aToB, manifoldContacts, numContacts, contactDist, tNormal);

		const PxU32 faceContacts = numContacts - originalContacts;
		if(faceContacts < 2)
		{
			const HullPolygonData& referencePolygon = polyData.mPolygons[getPolygonIndex(polyData, map, V3Neg(tNormal))];
			generatedContactsEEContacts(capsule, polyData,referencePolygon, map, aToB, manifoldContacts, numContacts, contactDist, tNormal);
		}
	}
	else
	{
		generatedFaceContacts(capsule, polyData, map, aToB, manifoldContacts, numContacts, contactDist, tNormal);

		const PxU32 faceContacts = numContacts - originalContacts;
		if(faceContacts < 2)
		{
			const PxReal lowerEps = toleranceLength * PCM_WITNESS_POINT_LOWER_EPS;
			const PxReal upperEps = toleranceLength * PCM_WITNESS_POINT_UPPER_EPS;
			const PxReal tolerance = PxClamp(margin, lowerEps, upperEps);

			const PxU32 featureIndex = getWitnessPolygonIndex(polyData, map, V3Neg(tNormal), closest, tolerance);
			const HullPolygonData& referencePolygon = polyData.mPolygons[featureIndex];
			generatedContactsEEContacts(capsule, polyData,referencePolygon, map, aToB, manifoldContacts, numContacts, contactDist, tNormal);
		}
	}

	normal = tNormal;

	return true;
}

//sphere is in the local space of polyData, we treate sphere as capsule
EHDI bool Gu::generateSphereFullContactManifold(const CapsuleV& capsule, const PolygonalData& polyData, const SupportLocal* map, PersistentContact* manifoldContacts, PxU32& numContacts,
	const FloatVArg contactDist, Vec3V& normal, bool doOverlapTest)
{
	if(doOverlapTest)
	{	
		FloatV minOverlap;
		//overwrite the normal with the sperating axis
		if(!testPolyDataAxis(capsule, polyData, map, contactDist, minOverlap, normal))
			return false;
	}

	const FloatV zero = FZero();
	FloatV tEnter = zero, tExit = zero;
	const FloatV inflatedRadius = FAdd(capsule.radius, contactDist);
	const Vec3V dir = V3Neg(normal);
	const Vec3V vertexSpaceDir = M33MulV3(map->shape2Vertex, dir);
	const Vec3V p0 = M33MulV3(map->shape2Vertex, capsule.p0);
	if(intersectRayPolyhedron(p0, vertexSpaceDir, polyData, tEnter, tExit) && FAllGrtrOrEq(inflatedRadius, tEnter))
	{
		//ML: for sphere, the contact point A is always zeroV in its local space
		manifoldContacts[numContacts].mLocalPointA = V3Zero();
		manifoldContacts[numContacts].mLocalPointB = V3ScaleAdd(dir, tEnter, capsule.p0);
		manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(normal), tEnter);
	}

	return true;
}

//capsule is in the shape space of polyData 
EHDI bool Gu::computeMTD(const CapsuleV& capsule, const PolygonalData& polyData, const SupportLocal* map, FloatV& penDepth, Vec3V& normal)
{
	const FloatV contactDist = FZero();
	Vec3V separatingAxis;
	FloatV minOverlap;
	if(!testSATCapsulePoly(capsule, polyData, map, contactDist, minOverlap, separatingAxis))
		return false;

	//transform normal in to the worl space
	normal = map->transform.rotate(separatingAxis);
	penDepth = minOverlap;
		
	return true;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactBoxBox.cpp
using namespace px;
using namespace Gu;
using namespace aos;

EHDI static void getIncidentPolygon(Vec3V* pts, Vec3V& faceNormal, const Vec3VArg axis, const PxMatTransformV& transf1To0, const Vec3VArg extents)
{
	const FloatV zero = FZero();

	FloatV ex = V3GetX(extents);
	FloatV ey = V3GetY(extents);
	FloatV ez = V3GetZ(extents);

	const Vec3V u0 = transf1To0.getCol0();
	const Vec3V u1 = transf1To0.getCol1();
	const Vec3V u2 = transf1To0.getCol2();

	//calculate the insident face for b
	const FloatV d0 = V3Dot(u0, axis);
	const FloatV d1 = V3Dot(u1, axis);
	const FloatV d2 = V3Dot(u2, axis);

	const FloatV absd0 = FAbs(d0);
	const FloatV absd1 = FAbs(d1);
	const FloatV absd2 = FAbs(d2);

	if(FAllGrtrOrEq(absd0, absd1) && FAllGrtrOrEq(absd0, absd2))
	{
		//the incident face is on u0
		const BoolV con = FIsGrtr(d0, zero);
		faceNormal = V3Sel(con, V3Neg(u0), u0);
		ex = FSel(con, FNeg(ex), ex);
		const Vec3V r0 = V3Scale(u0, ex);
		const Vec3V r1 = V3Scale(u1, ey);
		const Vec3V r2 = V3Scale(u2, ez);

		const Vec3V temp0 = V3Add(transf1To0.p, r0);
		const Vec3V temp1 = V3Add(r1, r2);
		const Vec3V temp2 = V3Sub(r1, r2);
			
		pts[0] = V3Add(temp0, temp1);	// (-x/x,  y,  z)
		pts[1] = V3Add(temp0, temp2);	// (-x/x,  y, -z)
		pts[2] = V3Sub(temp0, temp1);	// (-x/x, -y, -z)
		pts[3] = V3Sub(temp0, temp2);	// (-x/x, -y,  z)
	}
	else if(FAllGrtrOrEq(absd1, absd2))
	{
		//the incident face is on u1
		const BoolV con = FIsGrtr(d1, zero);
		faceNormal = V3Sel(con, V3Neg(u1), u1);
		ey = FSel(con, FNeg(ey), ey);
		const Vec3V r0 = V3Scale(u0, ex);
		const Vec3V r1 = V3Scale(u1, ey);
		const Vec3V r2 = V3Scale(u2, ez);

		const Vec3V temp0 = V3Add(transf1To0.p, r1);
		const Vec3V temp1 = V3Add(r0, r2);
		const Vec3V temp2 = V3Sub(r0, r2);
			
		pts[0] = V3Add(temp0, temp1);	// (x,  -y/y,  z)
		pts[1] = V3Add(temp0, temp2);	// (x,  -y/y, -z)
		pts[2] = V3Sub(temp0, temp1);	// (-x, -y/y, -z)
		pts[3] = V3Sub(temp0, temp2);	// (-x, -y/y,  z)
	}
	else
	{
		//the incident face is on u2
		const BoolV con = FIsGrtr(d2, zero);
		faceNormal = V3Sel(con, V3Neg(u2), u2);
		ez = FSel(con, FNeg(ez), ez);
		const Vec3V r0 = V3Scale(u0, ex);
		const Vec3V r1 = V3Scale(u1, ey);
		const Vec3V r2 = V3Scale(u2, ez);

		const Vec3V temp0 = V3Add(transf1To0.p, r2);
		const Vec3V temp1 = V3Add(r0, r1);
		const Vec3V temp2 = V3Sub(r0, r1);
			
		pts[0] = V3Add(temp0, temp1);	// ( x,   y,  z)
		pts[1] = V3Add(temp0, temp2);	// ( x,  -y,  z)
		pts[2] = V3Sub(temp0, temp1);	// (-x,  -y,  z)
		pts[3] = V3Sub(temp0, temp2);	// (-x,   y,  z)
	}
}

//p0 and p1 is in the local space of AABB
EHDI static bool intersectSegmentAABB(const Vec3VArg p0, const Vec3VArg d, const Vec3VArg max, const Vec3VArg min, FloatV& tmin, FloatV& tmax)
{
	const Vec3V eps = V3Load(1e-6f);
	const Vec3V absV = V3Abs(d);
	const FloatV one = FOne();
	const Vec3V zero = V3Zero();
	const Vec3V fMax = Vec3V_From_FloatV(FMax());

	FloatV tminf = FZero();
	FloatV tmaxf = one;
	const BoolV isParallel = V3IsGrtr(eps, absV);
	const BoolV isOutsideOfRange = BOr(V3IsGrtr(p0, max), V3IsGrtr(min, p0));
	//const BoolV isParallelAndOutOfRange = BAnd(isParallel, isOutsideOfRange);

	if(!BAllEqFFFF(BAnd(isParallel, isOutsideOfRange)))
		return false;

	const Vec3V odd = V3RecipFast(d);
	const Vec3V t1 = V3Sel(isParallel, zero, V3Mul(V3Sub(min, p0), odd));
	const Vec3V t2 = V3Sel(isParallel, fMax, V3Mul(V3Sub(max, p0), odd));
		
	const Vec3V tt1 = V3Min(t1, t2);
	const Vec3V tt2 = V3Max(t1, t2);

	const FloatV ft1 = V3ExtractMax(tt1);
	const FloatV ft2 = V3ExtractMin(tt2);

	tminf = FMax(ft1, tminf);
	tmaxf = FMin(tmaxf, ft2);

	tmin = tminf;
	tmax = tmaxf;

	const BoolV con0 = FIsGrtr(tminf, tmaxf);
	const BoolV con1 = FIsGrtr(tminf, one);
	const BoolV isNotIntersect = BOr(con0, con1);
	return BAllEqFFFF(isNotIntersect) == 1;
}

//pts, faceNormal and contact normal are in the local space of new space
EHDI static void calculateContacts(const FloatVArg extentX_, const FloatVArg extentY_, Vec3V* pts, const Vec3VArg incidentFaceNormalInNew, const Vec3VArg localNormal, PersistentContact* manifoldContacts, PxU32& numContacts, const FloatVArg contactDist)
{
	const FloatV zero = FZero();
	const FloatV max = FMax();
	const FloatV eps = FLoad(1.0001f);

	const FloatV extentX = FMul(extentX_, eps);
	const FloatV extentY = FMul(extentY_, eps);

	const FloatV nExtentX = FNeg(extentX);
	const FloatV nExtentY = FNeg(extentY);
		
	bool pPenetration[4];
	bool pArea[4];

	Vec3V bmin = V3Splat(max);
	Vec3V bmax = V3Neg(bmin);

	const Vec3V bound = V3Merge(extentX, extentY, max);

	//get the projection point of pts
	for(PxU32 i=0; i< 4; ++i)
	{
		bmin = V3Min(bmin, pts[i]);
		bmax = V3Max(bmax, pts[i]);
		const FloatV z = FNeg(V3GetZ(pts[i]));
		if(FAllGrtr(contactDist, z))
		{
			pPenetration[i] = true;

			const Vec3V absPt = V3Abs(pts[i]);
			const BoolV con = V3IsGrtrOrEq(bound, absPt);
			if(BAllEqTTTT(con))
			{
				pArea[i] = true;
				
				//Add the point to the manifold
				manifoldContacts[numContacts].mLocalPointA = V3SetZ(pts[i], zero); //transformNewTo0.transform(localPointA);
				manifoldContacts[numContacts].mLocalPointB = pts[i];//transform1ToNew.transformInv(pts[i]);
				manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), z);
			}
			else
			{
				pArea[i] = false;
			}
		}
		else
		{
			pPenetration[i] = false;
			pArea[i] = false;
		}
	}

	if(numContacts == 4)
		return;

	//if(pPenetration[0] && pPenetration[1] && pPenetration[2] && pPenetration[3])
	{
		//if(!pArea[0] || !pArea[1] || !pArea[2] || !pArea[3])
		{
			const FloatV denom = V3GetZ(incidentFaceNormalInNew);
			{
				const Vec3V q0 = V3Merge(extentX, extentY, zero);

				if(contains(pts, 4, q0, bmin, bmax))
				{
					const FloatV nom = V3Dot(incidentFaceNormalInNew, V3Sub(pts[0], q0));
					const FloatV t = FDiv(nom, denom);
					const FloatV pen = FNeg(t);
					if(FAllGrtr(contactDist, pen))
					{
						manifoldContacts[numContacts].mLocalPointA = q0;
						manifoldContacts[numContacts].mLocalPointB = V3SetZ(q0, t); 
						manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
					}
				}
			}
			
			{
				const Vec3V q0 = V3Merge(extentX, nExtentY, zero);
				if(contains(pts, 4, q0, bmin, bmax))
				{
					const FloatV nom = V3Dot(incidentFaceNormalInNew, V3Sub(pts[0], q0));
					const FloatV t = FDiv(nom, denom);
					const FloatV pen = FNeg(t);
					if(FAllGrtr(contactDist, pen))
					{
						manifoldContacts[numContacts].mLocalPointA = q0;
						manifoldContacts[numContacts].mLocalPointB = V3SetZ(q0, t);
						manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
					}
				}
			}
				
			{
				const Vec3V q0 = V3Merge( nExtentX, extentY, zero);
				if(contains(pts, 4, q0, bmin, bmax))
				{
					const FloatV nom = V3Dot(incidentFaceNormalInNew, V3Sub(pts[0], q0));
					const FloatV t = FDiv(nom, denom);
					const FloatV pen = FNeg(t);
					if(FAllGrtr(contactDist, pen))
					{
						manifoldContacts[numContacts].mLocalPointA = q0;
						manifoldContacts[numContacts].mLocalPointB = V3SetZ(q0, t);
						manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
					}
				}
			}
			
			{
				const Vec3V q0 = V3Merge(nExtentX, nExtentY, zero);
					
				if(contains(pts, 4, q0, bmin, bmax))
				{
					const FloatV nom = V3Dot(incidentFaceNormalInNew, V3Sub(pts[0], q0));
					const FloatV t = FDiv(nom, denom);
					const FloatV pen = FNeg(t);
					if(FAllGrtr(contactDist, pen))
					{
						manifoldContacts[numContacts].mLocalPointA = q0;
						manifoldContacts[numContacts].mLocalPointB = V3SetZ(q0, t);
						manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
					}
				}
			}
		}
	}

	const Vec3V ext = V3Merge(extentX, extentY, max);
	const Vec3V negExt = V3Merge(nExtentX, nExtentY, FNeg(FAdd(contactDist, FEps())));

	for (PxU32 rStart = 0, rEnd = 3; rStart < 4; rEnd = rStart++)
	{
		const Vec3V p0 = pts[rStart];	
		const Vec3V p1 = pts[rEnd];

		if(!pPenetration[rStart] &&  !pPenetration[rEnd])
			continue;

		const bool con0 = pPenetration[rStart] && pArea[rStart];
		const bool con1 = pPenetration[rEnd] && pArea[rEnd];
		if(con0 && con1)
			continue;

		//intersect t value with x plane
		const Vec3V p0p1 = V3Sub(p1, p0);

		FloatV tmin, tmax;
		if(intersectSegmentAABB(p0, p0p1, ext, negExt, tmin, tmax))
		{
			if(!con0)
			{
				const Vec3V intersectP = V3ScaleAdd(p0p1, tmin, p0);
				manifoldContacts[numContacts].mLocalPointA = V3SetZ(intersectP, zero);
				manifoldContacts[numContacts].mLocalPointB = intersectP;
				manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), FNeg(V3GetZ(intersectP)));
			}
			if(!con1)
			{
				const Vec3V intersectP = V3ScaleAdd(p0p1, tmax, p0);
				manifoldContacts[numContacts].mLocalPointA = V3SetZ(intersectP, zero);
				manifoldContacts[numContacts].mLocalPointB = intersectP;
				manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), FNeg(V3GetZ(intersectP)));
			}
		}
	}
}

EHDI static PxU32 doBoxBoxGenerateContacts(const Vec3VArg box0Extent, const Vec3VArg box1Extent, const PxMatTransformV& transform0, const PxMatTransformV& transform1, const FloatVArg contactDist, PersistentContact* manifoldContacts, PxU32& numContacts)
{
	const FloatV ea0 = V3GetX(box0Extent);
	const FloatV ea1 = V3GetY(box0Extent);
	const FloatV ea2 = V3GetZ(box0Extent);

	const FloatV eb0 = V3GetX(box1Extent);
	const FloatV eb1 = V3GetY(box1Extent);
	const FloatV eb2 = V3GetZ(box1Extent);

	const PxMatTransformV transform1To0 = transform0.transformInv(transform1);
	const Mat33V rot0To1 = M33Trnsps(transform1To0.rot);

	const Vec3V uEps = V3Load(1e-6f); 

	const FloatV zero = FZero();

	const FloatV tx = V3GetX(transform1To0.p);
	const FloatV ty = V3GetY(transform1To0.p);
	const FloatV tz = V3GetZ(transform1To0.p);
	const Vec3V col0 = transform1To0.getCol0();
	const Vec3V col1 = transform1To0.getCol1();
	const Vec3V col2 = transform1To0.getCol2();

	const Vec3V abs1To0Col0 = V3Add(V3Abs(col0), uEps);
	const Vec3V abs1To0Col1 = V3Add(V3Abs(col1), uEps);
	const Vec3V abs1To0Col2 = V3Add(V3Abs(col2), uEps);

	const Vec3V abs0To1Col0 = V3Add(V3Abs(rot0To1.col0), uEps);
	const Vec3V abs0To1Col1 = V3Add(V3Abs(rot0To1.col1), uEps);
	const Vec3V abs0To1Col2 = V3Add(V3Abs(rot0To1.col2), uEps);

	FloatV sign[6];
	FloatV overlap[6];

	FloatV ra, rb, radiusSum;
	//ua0
	{
		sign[0] = tx;

		const Vec3V vtemp3 = V3Mul(abs0To1Col0, box1Extent);
		rb = FAdd(V3GetX(vtemp3), FAdd(V3GetY(vtemp3), V3GetZ(vtemp3)));

		radiusSum = FAdd(ea0, rb);
		overlap[0] = FAdd(FSub(radiusSum, FAbs(sign[0])), contactDist);
		if(FAllGrtr(zero, overlap[0]))
			return false;
	}

	//ua1
	{
		sign[1] = ty;

		const Vec3V vtemp3 = V3Mul(abs0To1Col1, box1Extent);
		rb = FAdd(V3GetX(vtemp3), FAdd(V3GetY(vtemp3), V3GetZ(vtemp3)));

		radiusSum = FAdd(ea1, rb);
		overlap[1] = FAdd(FSub(radiusSum, FAbs(sign[1])), contactDist);
		if(FAllGrtr(zero, overlap[1]))
			return false;
	}

	//ua2	
	{
		sign[2] = tz;
		ra = ea2;

		const Vec3V vtemp3 = V3Mul(abs0To1Col2, box1Extent);
		rb = FAdd(V3GetX(vtemp3), FAdd(V3GetY(vtemp3), V3GetZ(vtemp3)));

		radiusSum = FAdd(ea2, rb);
		overlap[2] = FAdd(FSub(radiusSum, FAbs(sign[2])), contactDist);
		if(FAllGrtr(zero, overlap[2]))
			return false;
	}
   
	//ub0 
	{
		sign[3] = V3Dot(transform1To0.p, col0);

		const Vec3V vtemp3 = V3Mul(abs1To0Col0, box0Extent);
		ra = FAdd(V3GetX(vtemp3), FAdd(V3GetY(vtemp3), V3GetZ(vtemp3)));

		radiusSum = FAdd(ra, eb0);
		overlap[3] = FAdd(FSub(radiusSum, FAbs(sign[3])), contactDist);
		if(FAllGrtr(zero, overlap[3]))
			return false;
	}

	//ub1
	{
		sign[4] = V3Dot(transform1To0.p, col1);

		const Vec3V vtemp3 = V3Mul(abs1To0Col1, box0Extent);
		ra = FAdd(V3GetX(vtemp3), FAdd(V3GetY(vtemp3), V3GetZ(vtemp3)));

		radiusSum = FAdd(ra, eb1);
		overlap[4] = FAdd(FSub(radiusSum, FAbs(sign[4])), contactDist);
		if(FAllGrtr(zero, overlap[4]))
			return false;
	}

	//ub2
	{
		sign[5] = V3Dot(transform1To0.p, col2);

		const Vec3V vtemp3 = V3Mul(abs1To0Col2, box0Extent);
		ra = FAdd(V3GetX(vtemp3), FAdd(V3GetY(vtemp3), V3GetZ(vtemp3)));
		
		radiusSum = FAdd(ra, eb2);
		overlap[5] = FAdd(FSub(radiusSum, FAbs(sign[5])), contactDist);
		if(FAllGrtr(zero, overlap[5]))
			return false;
	}

	//ua0 X ub0
	{
		//B into A's space, ua0Xub0[0,-b3, b2]
		const FloatV absSign = FAbs(FSub(FMul(V3GetY(col0), tz), FMul(V3GetZ(col0), ty)));

		//B into A's space, ua0Xub0[0,-b3, b2]
		const FloatV vtemp0 = FMul(V3GetZ(abs1To0Col0), ea1);
		const FloatV vtemp1 = FMul(V3GetY(abs1To0Col0), ea2);
		ra = FAdd(vtemp0, vtemp1);

		//A into B's space, ua0Xub0[0, a3, -a2]
		const FloatV vtemp01 = FMul(V3GetZ(abs0To1Col0), eb1);
		const FloatV vtemp02 = FMul(V3GetY(abs0To1Col0), eb2);
		rb = FAdd(vtemp01, vtemp02);
		
		radiusSum = FAdd(FAdd(ra, rb), contactDist);
		if(FAllGrtr(absSign, radiusSum)) return false;
	}

	//ua0 X ub1
	{
		//B into A's space, ua0Xub0[0, -b3, b2]
		const FloatV absSign = FAbs(FSub(FMul(V3GetY(col1), tz), FMul(V3GetZ(col1), ty)));

		//B into A's space, ua0Xub0[0, -b3, b2]
		const FloatV vtemp0 = FMul(V3GetZ(abs1To0Col1), ea1);
		const FloatV vtemp1 = FMul(V3GetY(abs1To0Col1), ea2);
		ra = FAdd(vtemp0, vtemp1);

		//A into B's space, ua0Xub1[-a3, 0, a1]
		const FloatV vtemp01 = FMul(V3GetZ(abs0To1Col0), eb0);
		const FloatV vtemp02 = FMul(V3GetX(abs0To1Col0), eb2);
		rb = FAdd(vtemp01, vtemp02);
		
		radiusSum = FAdd(FAdd(ra, rb), contactDist);

		if(FAllGrtr(absSign, radiusSum)) return false;
	}

	//ua0 X ub2
	{
		//B into A's space, ua0Xub0[0, -b3, b2]
		const FloatV absSign = FAbs(FSub(FMul(V3GetY(col2), tz), FMul(V3GetZ(col2), ty)));

		//B into A's space, ua0Xub0[0, -b3, b2]
		const FloatV vtemp0 = FMul(V3GetZ(abs1To0Col2), ea1);
		const FloatV vtemp1 = FMul(V3GetY(abs1To0Col2), ea2);
		ra = FAdd(vtemp0, vtemp1);

		//A into B's space, ua0Xub1[a2, -a1, 0]
		const FloatV vtemp01 = FMul(V3GetY(abs0To1Col0), eb0);
		const FloatV vtemp02 = FMul(V3GetX(abs0To1Col0), eb1);
		rb = FAdd(vtemp01, vtemp02);
		
		radiusSum = FAdd(FAdd(ra, rb), contactDist);
		if(FAllGrtr(absSign, radiusSum)) return false;
	}

	//ua1 X ub0
	{
		//B into A's space, ua0Xub0[b3, 0, -b1]
		const FloatV absSign = FAbs(FSub(FMul(V3GetZ(col0), tx), FMul(V3GetX(col0), tz)));

		//B into A's space, ua0Xub0[b3, 0, -b1]
		const FloatV vtemp0 = FMul(V3GetZ(abs1To0Col0), ea0);
		const FloatV vtemp1 = FMul(V3GetX(abs1To0Col0), ea2);
		ra = FAdd(vtemp0, vtemp1);

		//A into B's space, ua0Xub1[0, a3, -a2]
		const FloatV vtemp01 = FMul(V3GetZ(abs0To1Col1), eb1);
		const FloatV vtemp02 = FMul(V3GetY(abs0To1Col1), eb2);
		rb = FAdd(vtemp01, vtemp02);
		
		radiusSum = FAdd(FAdd(ra, rb), contactDist);
		if(FAllGrtr(absSign, radiusSum))return false;
	}

	//ua1 X ub1
	{
		//B into A's space, ua0Xub0[b3, 0, -b1]
		const FloatV absSign = FAbs(FSub(FMul(V3GetZ(col1), tx), FMul(V3GetX(col1), tz)));

		//B into A's space, ua0Xub0[b3, 0, -b1]
		const FloatV vtemp0 = FMul(V3GetZ(abs1To0Col1), ea0);
		const FloatV vtemp1 = FMul(V3GetX(abs1To0Col1), ea2);
		ra = FAdd(vtemp0, vtemp1);

		//A into B's space, ua0Xub1[-a3, 0, -a1]
		const FloatV vtemp01 = FMul(V3GetZ(abs0To1Col1), eb0);
		const FloatV vtemp02 = FMul(V3GetX(abs0To1Col1), eb2);
		rb = FAdd(vtemp01, vtemp02);
		
		radiusSum = FAdd(FAdd(ra, rb), contactDist);
		if(FAllGrtr(absSign, radiusSum))return false;
	}

	//ua1 X ub2
	{
		//B into A's space, ua0Xub0[b3, 0, -b1]
		const FloatV absSign=FAbs(FSub(FMul(V3GetZ(col2), tx), FMul(V3GetX(col2), tz)));

		//B into A's space, ua0Xub0[b3, 0, -b1]
		const FloatV vtemp0 = FMul(V3GetZ(abs1To0Col2), ea0);
		const FloatV vtemp1 = FMul(V3GetX(abs1To0Col2), ea2);
		ra = FAdd(vtemp0, vtemp1);

		//A into B's space, ua0Xub1[a2, -a1, 0]
		const FloatV vtemp01 = FMul(V3GetY(abs0To1Col1), eb0);
		const FloatV vtemp02 = FMul(V3GetX(abs0To1Col1), eb1);
		rb = FAdd(vtemp01, vtemp02);
		
		radiusSum = FAdd(FAdd(ra, rb), contactDist);
		if(FAllGrtr(absSign, radiusSum))return false;
	}

	//ua2 X ub0
	{
		//B into A's space, ua2Xub0[-b2, b1, 0]
		const FloatV absSign = FAbs(FSub(FMul(V3GetX(col0), ty), FMul(V3GetY(col0), tx)));


		//B into A's space, ua2Xub0[-b2, b1, 0]
		const FloatV vtemp0 = FMul(V3GetY(abs1To0Col0), ea0);
		const FloatV vtemp1 = FMul(V3GetX(abs1To0Col0), ea1);
		ra = FAdd(vtemp0, vtemp1);

		//A into B's space, ua0Xub1[0, a3, -a2]
		const FloatV vtemp01 = FMul(V3GetZ(abs0To1Col2), eb1);
		const FloatV vtemp02 = FMul(V3GetY(abs0To1Col2), eb2);
		rb = FAdd(vtemp01, vtemp02);
		
		radiusSum = FAdd(FAdd(ra, rb), contactDist);
		if(FAllGrtr(absSign, radiusSum))return false;
	}

	//ua2 X ub1
	{
		//B into A's space, ua2Xub0[-b2, b1, 0]
		const FloatV absSign = FAbs(FSub(FMul(V3GetX(col1), ty), FMul(V3GetY(col1), tx)));

		//B into A's space, ua2Xub0[-b2, b1, 0]
		const FloatV vtemp0 = FMul(V3GetY(abs1To0Col1), ea0);
		const FloatV vtemp1 = FMul(V3GetX(abs1To0Col1), ea1);
		ra = FAdd(vtemp0, vtemp1);

		//A into B's space, ua0Xub1[-a3, 0, a1]
		const FloatV vtemp01 = FMul(V3GetZ(abs0To1Col2), eb0);
		const FloatV vtemp02 = FMul(V3GetX(abs0To1Col2), eb2);
		rb = FAdd(vtemp01, vtemp02);
		
		radiusSum = FAdd(FAdd(ra, rb), contactDist);
		if(FAllGrtr(absSign, radiusSum))return false;
	}

	//ua2 X ub2
	{
		//B into A's space, ua2Xub0[-b2, b1, 0]
		const FloatV absSign=FAbs(FSub(FMul(V3GetX(col2), ty), FMul(V3GetY(col2), tx)));

		//B into A's space, ua2Xub0[-b2, b1, 0]
		const FloatV vtemp0 = FMul(V3GetY(abs1To0Col2), ea0);
		const FloatV vtemp1 = FMul(V3GetX(abs1To0Col2), ea1);
		ra = FAdd(vtemp0, vtemp1);

		//A into B's space, ua0Xub1[a2, -a1, 0]
		const FloatV vtemp01 = FMul(V3GetY(abs0To1Col2), eb0);
		const FloatV vtemp02 = FMul(V3GetX(abs0To1Col2), eb1);
		rb = FAdd(vtemp01, vtemp02);
		
		radiusSum = FAdd(FAdd(ra, rb), contactDist);
		if(FAllGrtr(absSign, radiusSum))return false;
	}

	Vec3V mtd;

	PxU32 feature = 0;
	FloatV minOverlap = overlap[0];

	for(PxU32 i=1; i<6; ++i)
	{
		if(FAllGrtr(minOverlap, overlap[i]))
		{
			minOverlap = overlap[i];
			feature = i;
		}
	}

	PxMatTransformV newTransformV;
	const Vec3V axis00 = transform0.getCol0();
	const Vec3V axis01 = transform0.getCol1();
	const Vec3V axis02 = transform0.getCol2();
	const Vec3V axis10 = transform1.getCol0();
	const Vec3V axis11 = transform1.getCol1();
	const Vec3V axis12 = transform1.getCol2();

	Vec3V incidentFaceNormalInNew;
	Vec3V pts[4];
	bool flip = false;
	switch(feature)
	{
	case 0: //ua0
		{
			if(FAllGrtrOrEq(zero, sign[0]))	
			{
				mtd = axis00;
				newTransformV.rot.col0 = V3Neg(axis02);
				newTransformV.rot.col1 = axis01;
				newTransformV.rot.col2 = axis00;
				newTransformV.p = V3NegScaleSub(axis00, ea0, transform0.p);
			}
			else		
			{
				const Vec3V nAxis00 = V3Neg(axis00);
				mtd = nAxis00;
				newTransformV.rot.col0 = axis02;
				newTransformV.rot.col1 = axis01;
				newTransformV.rot.col2 = nAxis00;
				newTransformV.p = V3ScaleAdd(axis00, ea0, transform0.p);
			}
	
			const PxMatTransformV transform1ToNew = newTransformV.transformInv(transform1);
			const Vec3V localNormal = newTransformV.rotateInv(mtd);
			getIncidentPolygon(pts, incidentFaceNormalInNew, V3Neg(localNormal), transform1ToNew, box1Extent);
		
			calculateContacts(ea2, ea1, pts, incidentFaceNormalInNew, localNormal, manifoldContacts, numContacts, contactDist);
			
			break;
		};
	case 1: //ua1
		{
			if(FAllGrtrOrEq(zero, sign[1]))	
			{
				mtd = axis01;
				newTransformV.rot.col0 = axis00;
				newTransformV.rot.col1 = V3Neg(axis02);
				newTransformV.rot.col2 = axis01;	
				newTransformV.p = V3NegScaleSub(axis01, ea1, transform0.p); 
			}
			else		
			{
				const Vec3V nAxis01 = V3Neg(axis01);
				mtd = nAxis01;
				newTransformV.rot.col0 = axis00;
				newTransformV.rot.col1 = axis02;
				newTransformV.rot.col2 = nAxis01;	
				newTransformV.p = V3ScaleAdd(axis01, ea1, transform0.p);
			}

			const PxMatTransformV transform1ToNew = newTransformV.transformInv(transform1);
			const Vec3V localNormal = newTransformV.rotateInv(mtd);
			getIncidentPolygon(pts, incidentFaceNormalInNew, V3Neg(localNormal), transform1ToNew, box1Extent);

			calculateContacts(ea0, ea2, pts, incidentFaceNormalInNew, localNormal, manifoldContacts, numContacts, contactDist);

			break;
		};
	case 2: //ua2
		{
			if(FAllGrtrOrEq(zero, sign[2]))	
			{
				mtd = axis02;
				newTransformV.rot.col0 = axis00;
				newTransformV.rot.col1 = axis01;
				newTransformV.rot.col2 = axis02;	
				
				newTransformV.p = V3NegScaleSub(axis02, ea2, transform0.p); 
			}
			else		
			{
				const Vec3V nAxis02 = V3Neg(axis02);
				mtd = nAxis02;
				newTransformV.rot.col0 = axis00;
				newTransformV.rot.col1 = V3Neg(axis01);
				newTransformV.rot.col2 = nAxis02;
				newTransformV.p = V3ScaleAdd(axis02, ea2, transform0.p);
			}

			const PxMatTransformV transform1ToNew = newTransformV.transformInv(transform1);
			const Vec3V localNormal = newTransformV.rotateInv(mtd);
			getIncidentPolygon(pts, incidentFaceNormalInNew, V3Neg(localNormal), transform1ToNew, box1Extent);

			calculateContacts(ea0, ea1, pts, incidentFaceNormalInNew, localNormal, manifoldContacts, numContacts, contactDist);

			break;
		};
	case 3: //ub0
		{
			flip = true;
			if(FAllGrtrOrEq(zero, sign[3]))	
			{
				mtd = axis10;
				newTransformV.rot.col0 = axis12;
				newTransformV.rot.col1 = axis11;
				newTransformV.rot.col2 = V3Neg(axis10);
				newTransformV.p = V3ScaleAdd(axis10, eb0, transform1.p); //transform0.p - extents0.x*axis00;
			}
			else		
			{
				mtd = V3Neg(axis10);
				newTransformV.rot.col0 = V3Neg(axis12);
				newTransformV.rot.col1 = axis11;
				newTransformV.rot.col2 = axis10;
				newTransformV.p = V3NegScaleSub(axis10, eb0, transform1.p);//transform0.p + extents0.x*axis00;
			}

			const PxMatTransformV transform1ToNew = newTransformV.transformInv(transform0);
			const Vec3V localNormal = newTransformV.rotateInv(mtd);
			getIncidentPolygon(pts, incidentFaceNormalInNew, localNormal, transform1ToNew, box0Extent);

			calculateContacts(eb2, eb1, pts, incidentFaceNormalInNew, localNormal, manifoldContacts, numContacts, contactDist);
			
			break;
		};
	case 4: //ub1;
		{
			flip = true;
			if(FAllGrtrOrEq(zero, sign[4]))	
			{
				mtd = axis11;
				newTransformV.rot.col0 = axis10;
				newTransformV.rot.col1 = axis12;
				newTransformV.rot.col2 = V3Neg(axis11);
			
				newTransformV.p = V3ScaleAdd(axis11, eb1, transform1.p);
			}
			else		
			{
				mtd = V3Neg(axis11);

				newTransformV.rot.col0 = axis10;
				newTransformV.rot.col1 = V3Neg(axis12);
				newTransformV.rot.col2 = axis11;
				newTransformV.p = V3NegScaleSub(axis11, eb1, transform1.p); //transform0.p + extents0.x*axis00;
			}

			const PxMatTransformV transform1ToNew = newTransformV.transformInv(transform0);
			const Vec3V localNormal = newTransformV.rotateInv(mtd);
			getIncidentPolygon(pts, incidentFaceNormalInNew, localNormal, transform1ToNew, box0Extent);
			calculateContacts(eb0, eb2, pts, incidentFaceNormalInNew, localNormal, manifoldContacts, numContacts, contactDist);
			break;
		}
	case 5: //ub2;
		{

			flip = true;
			if(FAllGrtrOrEq(zero, sign[5]))	
			{
				mtd = axis12;
				newTransformV.rot.col0 = axis10;
				newTransformV.rot.col1 = V3Neg(axis11);
				newTransformV.rot.col2 = V3Neg(axis12);
				newTransformV.p = V3ScaleAdd(axis12, eb2, transform1.p);
			}
			else		
			{
				mtd = V3Neg(axis12);
				
				newTransformV.rot.col0 = axis10;
				newTransformV.rot.col1 = axis11;
				newTransformV.rot.col2 = axis12;
				newTransformV.p = V3NegScaleSub(axis12, eb2, transform1.p);
			}

			const PxMatTransformV transform1ToNew = newTransformV.transformInv(transform0);
			const Vec3V localNormal = newTransformV.rotateInv(mtd);
			getIncidentPolygon(pts, incidentFaceNormalInNew, localNormal, transform1ToNew, box0Extent);
		
			calculateContacts(eb0, eb1, pts, incidentFaceNormalInNew, localNormal, manifoldContacts, numContacts, contactDist);
			break;
		};
	default:
		return false;
	}

	if(numContacts != 0)
	{
		if(flip)
		{
			for(PxU32 i=0; i<numContacts; ++i)
			{
				const Vec3V localB = manifoldContacts[i].mLocalPointB;
				manifoldContacts[i].mLocalPointB = manifoldContacts[i].mLocalPointA;
				manifoldContacts[i].mLocalPointA = localB;
			}
		}
		const PxMatTransformV transformNewTo1 = transform1.transformInv(newTransformV);
		const PxMatTransformV transformNewTo0 = transform0.transformInv(newTransformV);
		//transform points to the local space of transform0 and transform1
		const Vec3V localNormalInB = transformNewTo1.rotate(Vec3V_From_Vec4V(manifoldContacts[0].mLocalNormalPen));

		for(PxU32 i=0; i<numContacts; ++i)
		{
			manifoldContacts[i].mLocalPointA = transformNewTo0.transform(manifoldContacts[i].mLocalPointA);
			manifoldContacts[i].mLocalPointB = transformNewTo1.transform(manifoldContacts[i].mLocalPointB);
			manifoldContacts[i].mLocalNormalPen = V4SetW(localNormalInB, V4GetW(manifoldContacts[i].mLocalNormalPen));
		}
	}

	return true;
}  

EHDI bool Gu::pcmContactBoxBox(GU_CONTACT_METHOD_ARGS)
{
	E_UNUSED(renderOutput);
	E_ASSERT(transform1.q.isSane());
	E_ASSERT(transform0.q.isSane());

	const PxBoxGeometry& shapeBox0 = checkedCast<PxBoxGeometry>(shape0);
	const PxBoxGeometry& shapeBox1 = checkedCast<PxBoxGeometry>(shape1);

	// Get actual shape data
	PersistentContactManifold& manifold = cache.getManifold();
	PxPrefetchLine(&manifold, 256);

	const FloatV contactDist = FLoad(params.mContactDistance);
	const Vec3V boxExtents0 = V3LoadU(shapeBox0.halfExtents);
	const Vec3V boxExtents1 = V3LoadU(shapeBox1.halfExtents);
	
	//Transfer A into the local space of B
	const PxTransformV transf0 = loadTransformA(transform0);
	const PxTransformV transf1 = loadTransformA(transform1);
	const PxTransformV curRTrans(transf1.transformInv(transf0));
	const PxMatTransformV aToB(curRTrans);

	const PxReal toleranceLength = params.mToleranceLength;
	const FloatV boxMargin0 = CalculatePCMBoxMargin(boxExtents0, toleranceLength);
	const FloatV boxMargin1 = CalculatePCMBoxMargin(boxExtents1, toleranceLength);
	const FloatV minMargin = FMin(boxMargin0, boxMargin1);

	const PxU32 initialContacts = manifold.mNumContacts;
	const FloatV projectBreakingThreshold = FMul(minMargin, FLoad(0.8f));
	
	manifold.refreshContactPoints(aToB, projectBreakingThreshold, contactDist);

	const PxU32 newContacts = manifold.mNumContacts;

	const bool bLostContacts = (newContacts != initialContacts);
  
	const FloatV radiusA = V3Length(boxExtents0);
	const FloatV radiusB = V3Length(boxExtents1);
	if(bLostContacts || manifold.invalidate_BoxConvex(curRTrans, transf0.q, transf1.q, minMargin, radiusA, radiusB))
	{
		manifold.setRelativeTransform(curRTrans, transf0.q, transf1.q);
		
		PxMatTransformV transfV0(transf0);
		PxMatTransformV transfV1(transf1);

		transfV0.rot.col0 = V3Normalize(transfV0.rot.col0);
		transfV0.rot.col1 = V3Normalize(transfV0.rot.col1);
		transfV0.rot.col2 = V3Normalize(transfV0.rot.col2);

		transfV1.rot.col0 = V3Normalize(transfV1.rot.col0);
		transfV1.rot.col1 = V3Normalize(transfV1.rot.col1);
		transfV1.rot.col2 = V3Normalize(transfV1.rot.col2);

		PersistentContact* manifoldContacts = EPX_CP_TO_PCP(contactBuffer.contacts);
		PxU32 numContacts = 0;
	
		if(doBoxBoxGenerateContacts(boxExtents0, boxExtents1, transfV0, transfV1, contactDist, manifoldContacts, numContacts)) 
		{
			if(numContacts > 0)
			{
				manifold.addBatchManifoldContacts(manifoldContacts, numContacts, toleranceLength);
				const Vec3V worldNormal = V3Normalize(transfV1.rotate(Vec3V_From_Vec4V(manifold.mContactPoints[0].mLocalNormalPen)));
				manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transfV1);
				return true;
			}
			else
			{
				const Vec3V zeroV = V3Zero();
				const BoxV box0(zeroV, boxExtents0);
				const BoxV box1(zeroV, boxExtents1);

				manifold.mNumWarmStartPoints = 0;
				const RelativeConvex<BoxV> convexA(box0, aToB);
				const LocalConvex<BoxV> convexB(box1);
				GjkOutput output;

				GjkStatus status = gjkPenetration<RelativeConvex<BoxV>, LocalConvex<BoxV> >(convexA, convexB, aToB.p, contactDist, true,
					manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);

				if(status == EPA_CONTACT)
				{			
					const RelativeConvex<BoxV> convexA1(box0, aToB);
					const LocalConvex<BoxV> convexB1(box1);

					status = epaPenetration(convexA1, convexB1, manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, 
						true, FLoad(toleranceLength), output);
				}

				if(status == GJK_CONTACT || status == EPA_CONTACT)
				{
					const FloatV replaceBreakingThreshold = FMul(minMargin, FLoad(0.05f));
					const Vec3V localPointA = aToB.transformInv(output.closestA);//curRTrans.transformInv(closestA);
					const Vec3V localPointB = output.closestB;
					const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(output.normal), output.penDep);
					numContacts += manifold.addManifoldPoint(localPointA, localPointB, localNormalPen, replaceBreakingThreshold);

					//transform the normal back to world space
					const Vec3V worldNormal = V3Normalize(transf1.rotate(output.normal));

					manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transf1, contactDist);

					return true;
				}
			}
		}
	}
	else if(manifold.getNumContacts() > 0)
	{
		const Vec3V worldNormal = manifold.getWorldNormal(transf1);
		manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transf1, contactDist);
		return true;
	}
	return false;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactBoxConvex.cpp
#define PCM_BOX_HULL_DEBUG 0

using namespace px;
using namespace Gu;
using namespace aos;

EHDI static bool fullContactsGenerationBoxConvex(const GjkConvex* relativeConvex, const GjkConvex* localConvex, const PxTransformV& transf0, const PxTransformV& transf1,
											PersistentContact* manifoldContacts, PxContactBuffer& contactBuffer, PersistentContactManifold& manifold, Vec3VArg normal, 
											const Vec3VArg closestA, const Vec3VArg closestB, const FloatVArg contactDist, bool idtScale, bool doOverlapTest, PxRenderOutput* renderOutput,
											PxReal toleranceLength)
{
	PolygonalData polyData0;

	const BoxV& box = relativeConvex->getConvex<BoxV>();
	const ConvexHullV& convexHull = localConvex->getConvex<ConvexHullV>();

	PxVec3 halfExtents;
	V3StoreU(box.extents, halfExtents);
	PCMPolygonalBox polyBox0(halfExtents);
	polyBox0.getPolygonalData(&polyData0);
	polyData0.mPolygonVertexRefs = gPCMBoxPolygonData_p();
	
	PolygonalData polyData1;
	getPCMConvexData(convexHull, idtScale, polyData1);
	
	const Mat33V identity = M33Identity();
	SupportLocalImpl<BoxV> map0(box, transf0, identity, identity, true);

	alignas(16) PxU8 buff1[sizeof(SupportLocalImpl<ConvexHullV>)];

	SupportLocal* map1 = (idtScale ? static_cast<SupportLocal*>(E_PLACEMENT_NEW(buff1, SupportLocalImpl<ConvexHullNoScaleV>)(static_cast<const ConvexHullNoScaleV&>(convexHull), transf1, convexHull.vertex2Shape, convexHull.shape2Vertex, idtScale)) : 
		static_cast<SupportLocal*>(E_PLACEMENT_NEW(buff1, SupportLocalImpl<ConvexHullV>)(convexHull, transf1, convexHull.vertex2Shape, convexHull.shape2Vertex, idtScale)));

	PxU32 numContacts = 0;
	if(generateFullContactManifold(polyData0, polyData1, &map0, map1, manifoldContacts, numContacts, contactDist, normal, closestA, closestB, box.getMarginF(), convexHull.getMarginF(), 
		doOverlapTest, renderOutput, toleranceLength))
	{
		if (numContacts > 0)
		{
			//reduce contacts
			manifold.addBatchManifoldContacts(manifoldContacts, numContacts, toleranceLength);

			const Vec3V worldNormal = manifold.getWorldNormal(transf1);

			manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transf1, contactDist);
		}
		else
		{
			//if doOverlapTest is true, which means GJK/EPA degenerate so we won't have any contact in the manifoldContacts array
			if (!doOverlapTest)
			{
				const Vec3V worldNormal = manifold.getWorldNormal(transf1);

				manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transf1, contactDist);
			}
		}
		
		return true;
	}
	return false;
}

EHDI static bool generateOrProcessContactsBoxConvex(	const GjkConvex* relativeConvex, const GjkConvex* localConvex, const PxTransformV& transf0, const PxTransformV& transf1, 
												const PxMatTransformV& aToB, GjkStatus status, GjkOutput& output, PersistentContactManifold& manifold, PxContactBuffer& contactBuffer, 
												PxU32 initialContacts, const FloatV minMargin, const FloatV contactDist,
												bool idtScale, PxReal toleranceLength, PxRenderOutput* renderOutput)
{
	if (status == GJK_NON_INTERSECT)
	{
		return false;
	}
	else
	{
		PersistentContact* manifoldContacts = EPX_CP_TO_PCP(contactBuffer.contacts);

		const Vec3V localNor = manifold.mNumContacts ? manifold.getLocalNormal() : V3Zero();

		const FloatV replaceBreakingThreshold = FMul(minMargin, FLoad(0.05f));

		//addGJKEPAContacts will increase the number of contacts in manifold. If status == EPA_CONTACT, we need to run epa algorithm and generate closest points, normal and
		//pentration. If epa doesn't degenerate, we will store the contacts information in the manifold. Otherwise, we will return true to do the fallback test
		const bool doOverlapTest = addGJKEPAContacts(relativeConvex, localConvex, aToB, status, manifoldContacts, replaceBreakingThreshold, FLoad(toleranceLength), output, manifold);

		//ML: after we refresh the contacts(newContacts) and generate a GJK/EPA contacts(we will store that in the manifold), if the number of contacts is still less than the original contacts,
		//which means we lose too mang contacts and we should regenerate all the contacts in the current configuration
		//Also, we need to look at the existing contacts, if the existing contacts has very different normal than the GJK/EPA contacts,
		//which means we should throw away the existing contacts and do full contact gen
		const bool fullContactGen = FAllGrtr(FLoad(0.707106781f), V3Dot(localNor, output.normal)) || (manifold.mNumContacts < initialContacts);

		if (fullContactGen || doOverlapTest)
		{
			return fullContactsGenerationBoxConvex(relativeConvex, localConvex, transf0, transf1, manifoldContacts, contactBuffer,
				manifold, output.normal, output.closestA, output.closestB, contactDist, idtScale, doOverlapTest, renderOutput, toleranceLength);
		}
		else
		{
			const Vec3V newLocalNor = V3Add(localNor, output.normal);
			const Vec3V worldNormal = V3Normalize(transf1.rotate(newLocalNor));
			//const Vec3V worldNormal = transf1.rotate(normal);
			manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transf1, contactDist);
			return true;
		}
	}
}

EHDI bool Gu::pcmContactBoxConvex(GU_CONTACT_METHOD_ARGS)
{
	E_ASSERT(transform1.q.isSane());
	E_ASSERT(transform0.q.isSane());

	const PxConvexMeshGeometry& shapeConvex = checkedCast<PxConvexMeshGeometry>(shape1);
	const PxBoxGeometry& shapeBox = checkedCast<PxBoxGeometry>(shape0);
	
	PersistentContactManifold& manifold = cache.getManifold();
	const ConvexHullData* hullData = _getHullData(shapeConvex);
	PxPrefetchLine(hullData);
	
	const FloatV contactDist = FLoad(params.mContactDistance);
	const Vec3V boxExtents = V3LoadU(shapeBox.halfExtents);

	const Vec3V vScale = V3LoadU_SafeReadW(shapeConvex.scale.scale);	// PT: safe because 'rotation' follows 'scale' in PxMeshScale
	
	const PxTransformV transf0 = loadTransformA(transform0);
	const PxTransformV transf1 = loadTransformA(transform1);
	const PxTransformV curRTrans(transf1.transformInv(transf0));
	const PxMatTransformV aToB(curRTrans);

	const PxReal toleranceLength = params.mToleranceLength;
	const FloatV convexMargin = CalculatePCMConvexMargin(hullData, vScale, toleranceLength);
	const FloatV boxMargin = CalculatePCMBoxMargin(boxExtents, toleranceLength);


	const FloatV minMargin = FMin(convexMargin, boxMargin);//FMin(boxMargin, convexMargin);
	const FloatV projectBreakingThreshold = FMul(minMargin, FLoad(0.8f));
	const PxU32 initialContacts = manifold.mNumContacts;

	manifold.refreshContactPoints(aToB, projectBreakingThreshold, contactDist);  

	const Vec3V extents = V3Mul(V3LoadU_SafeReadW(hullData->mInternal.mInternalExtents), vScale);
	const FloatV radiusA = V3Length(boxExtents);
	const FloatV radiusB = V3Length(extents);
	
	//After the refresh contact points, the numcontacts in the manifold will be changed
	const bool bLostContacts = (manifold.mNumContacts != initialContacts);

	if(bLostContacts || manifold.invalidate_BoxConvex(curRTrans, transf0.q, transf1.q, minMargin, radiusA, radiusB))
	{
		manifold.setRelativeTransform(curRTrans, transf0.q, transf1.q);
	
		GjkStatus status = manifold.mNumContacts > 0 ? GJK_UNDEFINED : GJK_NON_INTERSECT;

		const QuatV vQuat = QuatVLoadU(&shapeConvex.scale.rotation.x);
		const bool idtScale = shapeConvex.scale.isIdentity();
		const ConvexHullV convexHull(hullData, V3LoadU(hullData->mCenterOfMass), vScale, vQuat, idtScale);
		const BoxV box(V3Zero(), boxExtents);
		GjkOutput output;
		
		const RelativeConvex<BoxV> relativeConvex(box, aToB);

		if(idtScale)
		{
			const LocalConvex<ConvexHullNoScaleV> localConvex(static_cast<const ConvexHullNoScaleV&>(convexHull));
			
			status = gjkPenetration<RelativeConvex<BoxV>, LocalConvex<ConvexHullNoScaleV> >(relativeConvex, localConvex, aToB.p, contactDist, true,
				manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);

			return generateOrProcessContactsBoxConvex(&relativeConvex, &localConvex, transf0, transf1, aToB,
				status, output, manifold, contactBuffer, initialContacts,
				minMargin, contactDist, idtScale, toleranceLength, renderOutput);	
		}
		else
		{
			const LocalConvex<ConvexHullV> localConvex(convexHull);

			status = gjkPenetration<RelativeConvex<BoxV>, LocalConvex<ConvexHullV> >(relativeConvex, localConvex, aToB.p, contactDist, true,
				manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);

			return generateOrProcessContactsBoxConvex(&relativeConvex, &localConvex, transf0, transf1, aToB,
				status, output, manifold, contactBuffer, initialContacts,
				minMargin, contactDist, idtScale, toleranceLength, renderOutput);
		} 
	}
	else if(manifold.getNumContacts()>0)
	{
		const Vec3V worldNormal = manifold.getWorldNormal(transf1);
		manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, transf1, contactDist);
		return true;
	}

	return false;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactCapsuleBox.cpp
using namespace px;
using namespace Gu;
using namespace aos;

EHDI static bool fullContactsGenerationCapsuleBox(const CapsuleV& capsule, const BoxV& box, const PxVec3& halfExtents, const PxMatTransformV& aToB, const PxTransformV& transf0, const PxTransformV& transf1,
											PersistentContact* manifoldContacts, PxU32& numContacts, PxContactBuffer& contactBuffer, PersistentContactManifold& manifold, Vec3V& normal, const Vec3VArg closest,
											PxReal boxMargin, const FloatVArg contactDist, bool doOverlapTest, PxReal toleranceScale, PxRenderOutput* renderOutput)
{
	PolygonalData polyData;
	PCMPolygonalBox polyBox(halfExtents);
	polyBox.getPolygonalData(&polyData);

	const Mat33V identity = M33Identity();
	SupportLocalImpl<BoxV> map(box, transf1, identity, identity);

	PxU32 origContacts = numContacts;
	if(!generateCapsuleBoxFullContactManifold(capsule, polyData, &map, aToB, manifoldContacts, numContacts, contactDist, normal, closest, boxMargin, doOverlapTest, toleranceScale, renderOutput))
		return false;

	//EPA has contacts and we have new contacts, we discard the EPA contacts
	if(origContacts != 0 && numContacts != origContacts)
	{
		numContacts--;
		manifoldContacts++;
	}

	manifold.addBatchManifoldContacts2(manifoldContacts, numContacts);
		
	normal = transf1.rotate(normal);
		
	manifold.addManifoldContactsToContactBuffer(contactBuffer, normal, normal, transf0, capsule.radius, contactDist);
	
	return true;	
}

EHDI bool Gu::pcmContactCapsuleBox(GU_CONTACT_METHOD_ARGS)
{
	E_UNUSED(renderOutput);
	E_ASSERT(transform1.q.isSane());
	E_ASSERT(transform0.q.isSane());

	const PxCapsuleGeometry& shapeCapsule = checkedCast<PxCapsuleGeometry>(shape0);
	const PxBoxGeometry& shapeBox = checkedCast<PxBoxGeometry>(shape1);

	PersistentContactManifold& manifold = cache.getManifold();
	PxPrefetchLine(&manifold, 256);

	const Vec3V boxExtents = V3LoadU(shapeBox.halfExtents);

	const FloatV contactDist = FLoad(params.mContactDistance);

	const PxTransformV transf0 = loadTransformA(transform0);
	const PxTransformV transf1 = loadTransformA(transform1);

	const PxTransformV curRTrans = transf1.transformInv(transf0);
	const PxMatTransformV aToB_(curRTrans);

	const FloatV capsuleRadius = FLoad(shapeCapsule.radius);
	const FloatV capsuleHalfHeight = FLoad(shapeCapsule.halfHeight);

	const PxU32 initialContacts = manifold.mNumContacts;

	const PxReal toleranceLength = params.mToleranceLength;
	const FloatV boxMargin = CalculatePCMBoxMargin(boxExtents, toleranceLength);
	
	const FloatV minMargin = FMin(boxMargin, capsuleRadius);

	const FloatV projectBreakingThreshold = FMul(minMargin, FLoad(0.8f));
	
	const FloatV refreshDist = FAdd(contactDist, capsuleRadius);
	//refreshContactPoints remove invalid contacts from the manifold and update the number correspondingly
	manifold.refreshContactPoints(aToB_, projectBreakingThreshold, refreshDist);

	const bool bLostContacts = (manifold.mNumContacts != initialContacts);

	if(bLostContacts || manifold.invalidate_SphereCapsule(curRTrans, minMargin))	
	{
		GjkStatus status = manifold.mNumContacts > 0 ? GJK_UNDEFINED : GJK_NON_INTERSECT;

		manifold.setRelativeTransform(curRTrans);
		const PxMatTransformV aToB(curRTrans);
		
		const BoxV box(transf1.p, boxExtents);
		//transform capsule into the local space of box
		const CapsuleV capsule(aToB.p, aToB.rotate(V3Scale(V3UnitX(), capsuleHalfHeight)), capsuleRadius);
		const LocalConvex<CapsuleV> convexA(capsule);
		const LocalConvex<BoxV> convexB(box);
		GjkOutput output;

		const Vec3V initialSearchDir = V3Sub(capsule.getCenter(), box.getCenter());
		status = gjkPenetration<LocalConvex<CapsuleV>, LocalConvex<BoxV> >(convexA, convexB, initialSearchDir, contactDist, true, 
			manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);

		PersistentContact* manifoldContacts = EPX_CP_TO_PCP(contactBuffer.contacts);
		PxU32 numContacts = 0;
		bool doOverlapTest = false;
		if(status == GJK_NON_INTERSECT)
		{
			return false;
		}
		else if(status == GJK_DEGENERATE)
		{
			return fullContactsGenerationCapsuleBox(capsule, box, shapeBox.halfExtents, aToB, transf0, transf1, manifoldContacts, numContacts, contactBuffer,
				manifold, output.normal, output.closestB, box.getMarginF(), contactDist, true, params.mToleranceLength, renderOutput);
		}
		else 
		{
			if(status == GJK_CONTACT)
			{
				const Vec3V localPointA = aToB.transformInv(output.closestA);//curRTrans.transformInv(closestA);
				const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(output.normal), output.penDep);
				//Add contact to contact stream
				manifoldContacts[numContacts].mLocalPointA = localPointA;
				manifoldContacts[numContacts].mLocalPointB = output.closestB;
				manifoldContacts[numContacts++].mLocalNormalPen = localNormalPen;
			}
			else
			{
				E_ASSERT(status == EPA_CONTACT);
	
				status = epaPenetration(convexA, convexB, manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, 
					 true, FLoad(toleranceLength), output);
				if(status == EPA_CONTACT)
				{
					const Vec3V localPointA = aToB.transformInv(output.closestA);//curRTrans.transformInv(closestA);
					const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(output.normal), output.penDep);
					//Add contact to contact stream
					manifoldContacts[numContacts].mLocalPointA = localPointA;
					manifoldContacts[numContacts].mLocalPointB = output.closestB;
					manifoldContacts[numContacts++].mLocalNormalPen = localNormalPen;
				}
				else
				{
					doOverlapTest = true;
				}
			}
			
			if(initialContacts == 0 || bLostContacts || doOverlapTest)
			{
				return fullContactsGenerationCapsuleBox(capsule, box, shapeBox.halfExtents, aToB, transf0, transf1, manifoldContacts, numContacts, contactBuffer, manifold, output.normal, 
					output.closestB, box.getMarginF(), contactDist, doOverlapTest, params.mToleranceLength, renderOutput);
			}
			else
			{
				//The contacts is either come from GJK or EPA
				const FloatV replaceBreakingThreshold = FMul(minMargin, FLoad(0.1f));
				const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(output.normal), output.penDep);
				manifold.addManifoldPoint2(curRTrans.transformInv(output.closestA), output.closestB, localNormalPen, replaceBreakingThreshold);
				
				const Vec3V normal = transf1.rotate(output.normal);
				manifold.addManifoldContactsToContactBuffer(contactBuffer, normal, normal, transf0, capsuleRadius, contactDist);
			
				return true;
			}
		}	
	}
	else if(manifold.getNumContacts() > 0)
	{
		const Vec3V worldNormal = manifold.getWorldNormal(transf1);
		manifold.addManifoldContactsToContactBuffer(contactBuffer, worldNormal, worldNormal, transf0, capsuleRadius, contactDist);
		return true;
	}

	return false;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactCapsuleCapsule.cpp
using namespace px;
using namespace Gu;
using namespace aos;

EHDI static Vec4V pcmDistancePointSegmentTValue22(	const Vec3VArg a0, const Vec3VArg b0, 
												const Vec3VArg a1, const Vec3VArg b1,
												const Vec3VArg p0, const Vec3VArg p1,
												const Vec3VArg p2, const Vec3VArg p3)
{
	const Vec4V zero = V4Zero();
	const Vec3V ap00 = V3Sub(p0, a0);
	const Vec3V ap10 = V3Sub(p1, a0);
	const Vec3V ap01 = V3Sub(p2, a1);
	const Vec3V ap11 = V3Sub(p3, a1);

	const Vec3V ab0 = V3Sub(b0, a0);
	const Vec3V ab1 = V3Sub(b1, a1);

/*	const FloatV nom00 = V3Dot(ap00, ab0);
	const FloatV nom10 = V3Dot(ap10, ab0);
	const FloatV nom01 = V3Dot(ap01, ab1);
	const FloatV nom11 = V3Dot(ap11, ab1);*/

	const Vec4V combinedDot = V3Dot4(ap00, ab0, ap10, ab0, ap01, ab1, ap11, ab1);
	const FloatV nom00 = V4GetX(combinedDot);
	const FloatV nom10 = V4GetY(combinedDot);
	const FloatV nom01 = V4GetZ(combinedDot);
	const FloatV nom11 = V4GetW(combinedDot);
	
	const FloatV denom0 = V3Dot(ab0, ab0);
	const FloatV denom1 = V3Dot(ab1, ab1);

	const Vec4V nom = V4Merge(nom00, nom10, nom01, nom11);
	const Vec4V denom = V4Merge(denom0, denom0, denom1, denom1);

	const Vec4V tValue = V4Div(nom, denom);
	return V4Sel(V4IsEq(denom, zero), zero, tValue);
}

EHDI static void storeContact(const Vec3VArg contact, const Vec3VArg normal, const FloatVArg separation, PxContactBuffer& buffer)
{
	outputSimplePCMContact(buffer, contact, normal, separation);
}

EHDI bool Gu::pcmContactCapsuleCapsule(GU_CONTACT_METHOD_ARGS)
{
	E_UNUSED(renderOutput);
	E_UNUSED(cache);
	E_ASSERT(transform1.q.isSane());
	E_ASSERT(transform0.q.isSane());

	// Get actual shape data
	const PxCapsuleGeometry& shapeCapsule0 = checkedCast<PxCapsuleGeometry>(shape0);
	const PxCapsuleGeometry& shapeCapsule1 = checkedCast<PxCapsuleGeometry>(shape1);

	const Vec3V _p0 = V3LoadA(&transform0.p.x);
	const QuatV q0 = QuatVLoadA(&transform0.q.x);

	const Vec3V _p1 = V3LoadA(&transform1.p.x);
	const QuatV q1 = QuatVLoadA(&transform1.q.x);

	/*PsTransformV transf0(p0, q0);
	PsTransformV transf1(p1, q1);*/
	
	const FloatV r0 = FLoad(shapeCapsule0.radius);
	const FloatV halfHeight0 = FLoad(shapeCapsule0.halfHeight);

	const FloatV r1 = FLoad(shapeCapsule1.radius);
	const FloatV halfHeight1 = FLoad(shapeCapsule1.halfHeight);

	const FloatV cDist = FLoad(params.mContactDistance);

	const Vec3V positionOffset = V3Scale(V3Add(_p0, _p1), FHalf());
	const Vec3V p0 = V3Sub(_p0, positionOffset);
	const Vec3V p1 = V3Sub(_p1, positionOffset);

	const FloatV zero = FZero();
	//const FloatV one = FOne();
	const Vec3V zeroV = V3Zero();

	/*const Vec3V positionOffset = V3Scale(V3Add(transf0.p, transf1.p), FloatV_From_F32(0.5f));
	transf0.p = V3Sub(transf0.p, positionOffset);
	transf1.p = V3Sub(transf1.p, positionOffset);*/

	const Vec3V basisVector0 = QuatGetBasisVector0(q0);
	const Vec3V tmp0 = V3Scale(basisVector0, halfHeight0);
	const Vec3V s0 = V3Add(p0, tmp0);
	const Vec3V e0 = V3Sub(p0, tmp0);
	const Vec3V d0 = V3Sub(e0, s0);

	const Vec3V basisVector1 = QuatGetBasisVector0(q1);
	const Vec3V tmp1 = V3Scale(basisVector1, halfHeight1);
	const Vec3V s1 = V3Add(p1, tmp1);
	const Vec3V e1 = V3Sub(p1, tmp1); 

	const Vec3V d1 = V3Sub(e1, s1);

	const FloatV sumRadius = FAdd(r0, r1);
	const FloatV inflatedSum = FAdd(sumRadius, cDist);
	const FloatV inflatedSumSquared = FMul(inflatedSum, inflatedSum);
	const FloatV a = V3Dot(d0, d0);//squared length of segment1
	const FloatV e = V3Dot(d1, d1);//squared length of segment2
	const FloatV eps = FLoad(1e-6f);//FEps();

	FloatV t0, t1;
	const FloatV sqDist0 = distanceSegmentSegmentSquared(s0, d0, s1, d1, t0, t1);

	if(FAllGrtrOrEq(inflatedSumSquared, sqDist0))
	{
		const Vec4V zeroV4 = V4Zero();
		const Vec4V oneV4 = V4One();
		//check to see whether these two capsule are paralle
		const FloatV parallelTolerance  = FLoad(0.9998f);
		
		const BoolV con0 = FIsGrtr(eps, a);
		const BoolV con1 = FIsGrtr(eps, e);
		const Vec3V dir0 = V3Sel(con0, zeroV, V3ScaleInv(d0, FSqrt(a)));
		const Vec3V dir1 = V3Sel(con1, zeroV, V3ScaleInv(d1, FSqrt(e)));

		const FloatV cos = FAbs(V3Dot(dir0, dir1));
		if(FAllGrtr(cos, parallelTolerance))//paralle
		{
			//project s, e into s1e1
			const Vec4V t = pcmDistancePointSegmentTValue22(s0, e0, s1, e1,
															s1, e1, s0, e0);

			const BoolV con = BAnd(V4IsGrtrOrEq(t, zeroV4), V4IsGrtrOrEq(oneV4, t));
			const BoolV con00 = BGetX(con);
			const BoolV con01 = BGetY(con);
			const BoolV con10 = BGetZ(con);
			const BoolV con11 = BGetW(con);

		/*	PxU32 conditions[4];
			F32Array_Aligned_From_Vec4V(con, (PxF32*)conditions);*/
		
			PxU32 numContact=0;

			if(BAllEqTTTT(con00))
			{
				const Vec3V projS1 = V3ScaleAdd(d0, V4GetX(t), s0);
				const Vec3V v = V3Sub(projS1, s1);
				const FloatV sqDist = V3Dot(v, v);
				const BoolV bCon = BAnd(FIsGrtr(sqDist, eps), FIsGrtr(inflatedSumSquared, sqDist));
				
				if(BAllEqTTTT(bCon))
				{
					const FloatV dist = FSqrt(sqDist);
					const FloatV pen = FSub(dist, sumRadius);
					const Vec3V normal = V3ScaleInv(v, dist);
					E_ASSERT(isFiniteVec3V(normal));
					const Vec3V _p = V3NegScaleSub(normal, r0, projS1);
					const Vec3V p = V3Add(_p, positionOffset);
					
					storeContact(p, normal, pen, contactBuffer);
					numContact++;
				}
			}
			if(BAllEqTTTT(con01))
			{
				const Vec3V projE1 = V3ScaleAdd(d0, V4GetY(t), s0);
				const Vec3V v = V3Sub(projE1, e1);
				const FloatV sqDist = V3Dot(v, v);
				const BoolV bCon = BAnd(FIsGrtr(sqDist, eps), FIsGrtr(inflatedSumSquared, sqDist));
				
				if(BAllEqTTTT(bCon))
				{
					const FloatV dist = FSqrt(sqDist);
					const FloatV pen = FSub(dist, sumRadius);
					const Vec3V normal = V3ScaleInv(v, dist);
					E_ASSERT(isFiniteVec3V(normal));
					const Vec3V _p = V3NegScaleSub(normal, r0, projE1);
					const Vec3V p = V3Add(_p, positionOffset);
					storeContact(p, normal, pen, contactBuffer);
					numContact++;
				}
			}

			if(BAllEqTTTT(con10))
			{
				const Vec3V projS0 = V3ScaleAdd(d1, V4GetZ(t), s1);
				const Vec3V v = V3Sub(s0, projS0);
				const FloatV sqDist = V3Dot(v, v);
				const BoolV bCon = BAnd(FIsGrtr(sqDist, eps), FIsGrtr(inflatedSumSquared, sqDist));
				
				if(BAllEqTTTT(bCon))
				{
					const FloatV dist = FSqrt(sqDist);
					const FloatV pen = FSub(dist, sumRadius);
					const Vec3V normal = V3ScaleInv(v, dist);
				 	E_ASSERT(isFiniteVec3V(normal));
					const Vec3V _p = V3NegScaleSub(normal, r0, s0);
					const Vec3V p = V3Add(_p, positionOffset);
					//const Vec3V p = V3ScaleAdd(normal, r0, s0);
					storeContact(p, normal, pen, contactBuffer);
					numContact++;
				}
			}

			if(BAllEqTTTT(con11))
			{
				const Vec3V projE0 = V3ScaleAdd(d1, V4GetW(t), s1);
				const Vec3V v = V3Sub(e0, projE0);
				const FloatV sqDist = V3Dot(v, v);
				const BoolV bCon = BAnd(FIsGrtr(sqDist, eps), FIsGrtr(inflatedSumSquared, sqDist));
				
				if(BAllEqTTTT(bCon))
				{
					const FloatV dist = FSqrt(sqDist);
					const FloatV pen = FSub(dist, sumRadius);
					const Vec3V normal = V3ScaleInv(v, dist);
					E_ASSERT(isFiniteVec3V(normal));
					const Vec3V _p = V3NegScaleSub(normal, r0, e0);
					const Vec3V p = V3Add(_p, positionOffset);
					//const Vec3V p = V3ScaleAdd(normal, r0, e0);
					storeContact(p, normal, pen, contactBuffer);
					numContact++;
				}
			}

			if(numContact)
				return true;
		}  

		const Vec3V closestA = V3ScaleAdd(d0, t0, s0);
		const Vec3V closestB = V3ScaleAdd(d1, t1, s1);
		
		const BoolV con = FIsGrtr(eps, sqDist0);
		//const Vec3V normal = V3Sel(FIsEq(dist, zero), V3Sel(FIsGrtr(a, eps), V3Normalise(d0), V3Scale(V3Sub(closestA, closestB), FRecip(dist)));
		const Vec3V _normal = V3Sel(con, V3Sel(FIsGrtr(a, eps), d0, V3UnitX()), V3Sub(closestA, closestB));
		const Vec3V normal = V3Normalize(_normal);
		E_ASSERT(isFiniteVec3V(normal));
		const Vec3V _point = V3NegScaleSub(normal, r0, closestA);
		const Vec3V p = V3Add(_point, positionOffset);
		const FloatV dist = FSel(con, zero, FSqrt(sqDist0));
		const FloatV pen = FSub(dist, sumRadius);
		//E_ASSERT(FAllGrtrOrEq(zero, pen));
		storeContact(p, normal, pen, contactBuffer);
		return true;
	}
	return false;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactCapsuleConvex.cpp
using namespace px;
using namespace Gu;
using namespace aos;

EHDI static bool fullContactsGenerationCapsuleConvex(const CapsuleV& capsule, const ConvexHullV& convexHull, const PxMatTransformV& aToB, const PxTransformV& transf0,const PxTransformV& transf1,
												PersistentContact* manifoldContacts, PxContactBuffer& contactBuffer, bool idtScale, PersistentContactManifold& manifold, Vec3VArg normal, 
												const Vec3VArg closest, PxReal tolerance, const FloatVArg contactDist, bool doOverlapTest, PxRenderOutput* renderOutput, PxReal toleranceLength)
{
	E_UNUSED(renderOutput);
	PolygonalData polyData;
	getPCMConvexData(convexHull,idtScale, polyData);

	alignas(16) PxU8 buff[sizeof(SupportLocalImpl<ConvexHullV>)];
	SupportLocal* map = (idtScale ? static_cast<SupportLocal*>(E_PLACEMENT_NEW(buff, SupportLocalImpl<ConvexHullNoScaleV>)(static_cast<const ConvexHullNoScaleV&>(convexHull), transf1, convexHull.vertex2Shape, convexHull.shape2Vertex, idtScale)) : 
	static_cast<SupportLocal*>(E_PLACEMENT_NEW(buff, SupportLocalImpl<ConvexHullV>)(convexHull, transf1, convexHull.vertex2Shape, convexHull.shape2Vertex, idtScale)));

	PxU32 numContacts = 0;
	if(!generateFullContactManifold(capsule, polyData, map, aToB, manifoldContacts, numContacts, contactDist, normal, closest, tolerance, doOverlapTest, toleranceLength))
		return false;

	if (numContacts > 0)
	{
		manifold.addBatchManifoldContacts2(manifoldContacts, numContacts);
		//transform normal into the world space
		normal = transf1.rotate(normal);
		manifold.addManifoldContactsToContactBuffer(contactBuffer, normal, normal, transf0, capsule.radius, contactDist);
	}
	else
	{
		if (!doOverlapTest)
		{
			normal = transf1.rotate(normal);
			manifold.addManifoldContactsToContactBuffer(contactBuffer, normal, normal, transf0, capsule.radius, contactDist);
		}
	}

	return true;
}

EHDI bool Gu::pcmContactCapsuleConvex(GU_CONTACT_METHOD_ARGS)
{
	E_UNUSED(renderOutput);
	E_ASSERT(transform1.q.isSane());
	E_ASSERT(transform0.q.isSane());

	const PxConvexMeshGeometry& shapeConvex = checkedCast<PxConvexMeshGeometry>(shape1);
	const PxCapsuleGeometry& shapeCapsule = checkedCast<PxCapsuleGeometry>(shape0);

	PersistentContactManifold& manifold = cache.getManifold();

	const ConvexHullData* hullData = _getHullData(shapeConvex);
	PxPrefetchLine(hullData);

	const Vec3V vScale = V3LoadU_SafeReadW(shapeConvex.scale.scale);	// PT: safe because 'rotation' follows 'scale' in PxMeshScale

	const FloatV contactDist = FLoad(params.mContactDistance);
	const FloatV capsuleHalfHeight = FLoad(shapeCapsule.halfHeight);
	const FloatV capsuleRadius = FLoad(shapeCapsule.radius);
	
	//Transfer A into the local space of B
	const PxTransformV transf0 = loadTransformA(transform0);
	const PxTransformV transf1 = loadTransformA(transform1);
	const PxTransformV curRTrans(transf1.transformInv(transf0));
	const PxMatTransformV aToB(curRTrans);
	
	const PxReal toleranceLength = params.mToleranceLength;
	const FloatV convexMargin = CalculatePCMConvexMargin(hullData, vScale, toleranceLength);
	const FloatV capsuleMinMargin = CalculateCapsuleMinMargin(capsuleRadius);
	const FloatV minMargin = FMin(convexMargin, capsuleMinMargin);
	
	const PxU32 initialContacts = manifold.mNumContacts;
	const FloatV projectBreakingThreshold = FMul(minMargin, FLoad(1.25f));
	const FloatV refreshDist = FAdd(contactDist, capsuleRadius);

	manifold.refreshContactPoints(aToB, projectBreakingThreshold, refreshDist);

	//ML: after refreshContactPoints, we might lose some contacts
	const bool bLostContacts = (manifold.mNumContacts != initialContacts);

	GjkStatus status = manifold.mNumContacts > 0 ? GJK_UNDEFINED : GJK_NON_INTERSECT;

	if(bLostContacts || manifold.invalidate_SphereCapsule(curRTrans, minMargin))
	{
		const bool idtScale = shapeConvex.scale.isIdentity();

		manifold.setRelativeTransform(curRTrans);
		const QuatV vQuat = QuatVLoadU(&shapeConvex.scale.rotation.x);  
		const ConvexHullV convexHull(hullData, V3LoadU(hullData->mCenterOfMass), vScale, vQuat, idtScale);
	
		//transform capsule(a) into the local space of convexHull(b)
		const CapsuleV capsule(aToB.p, aToB.rotate(V3Scale(V3UnitX(), capsuleHalfHeight)), capsuleRadius);
	
		GjkOutput output;
		const LocalConvex<CapsuleV> convexA(capsule);
		const Vec3V initialSearchDir = V3Sub(capsule.getCenter(), convexHull.getCenter());
		if(idtScale)
		{
			const LocalConvex<ConvexHullNoScaleV> convexB(*EPX_CONVEX_TO_NOSCALECONVEX(&convexHull));
			status = gjkPenetration<LocalConvex<CapsuleV>, LocalConvex<ConvexHullNoScaleV> >(convexA, convexB, initialSearchDir, contactDist, true,
				manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);
		}
		else
		{
			const LocalConvex<ConvexHullV> convexB(convexHull);
			status = gjkPenetration<LocalConvex<CapsuleV>, LocalConvex<ConvexHullV> >(convexA, convexB, initialSearchDir, contactDist, true,
				manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);
		}

		PersistentContact* manifoldContacts = EPX_CP_TO_PCP(contactBuffer.contacts);
		bool doOverlapTest = false;
		if(status == GJK_NON_INTERSECT)
		{
			return false;
		}
		else if(status == GJK_DEGENERATE)
		{
			return fullContactsGenerationCapsuleConvex(capsule, convexHull, aToB, transf0, transf1, manifoldContacts, contactBuffer, idtScale, manifold, output.normal, 
				output.closestB, convexHull.getMarginF(), contactDist, true, renderOutput, toleranceLength);
		}
		else 
		{
			const FloatV replaceBreakingThreshold = FMul(minMargin, FLoad(0.05f));

			if(status == GJK_CONTACT)
			{
				const Vec3V localPointA = aToB.transformInv(output.closestA);//curRTrans.transformInv(closestA);
				const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(output.normal), output.penDep);
				//Add contact to contact stream
				manifoldContacts[0].mLocalPointA = localPointA;
				manifoldContacts[0].mLocalPointB = output.closestB;
				manifoldContacts[0].mLocalNormalPen = localNormalPen;

				//Add contact to manifold
				manifold.addManifoldPoint2(localPointA, output.closestB, localNormalPen, replaceBreakingThreshold);
			}
			else
			{
				E_ASSERT(status == EPA_CONTACT);
				
				if(idtScale)
				{
					const LocalConvex<ConvexHullNoScaleV> convexB(*EPX_CONVEX_TO_NOSCALECONVEX(&convexHull));
					status = epaPenetration(convexA, convexB, manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints,
					 true, FLoad(toleranceLength), output);
				}
				else
				{
					const LocalConvex<ConvexHullV> convexB(convexHull);
					status = epaPenetration(convexA, convexB, manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints,
					true, FLoad(toleranceLength), output);
				}
				
				if(status == EPA_CONTACT)
				{
					const Vec3V localPointA = aToB.transformInv(output.closestA);//curRTrans.transformInv(closestA);
					const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(output.normal), output.penDep);
					//Add contact to contact stream
					manifoldContacts[0].mLocalPointA = localPointA;
					manifoldContacts[0].mLocalPointB = output.closestB;
					manifoldContacts[0].mLocalNormalPen = localNormalPen;

					//Add contact to manifold
					manifold.addManifoldPoint2(localPointA, output.closestB, localNormalPen, replaceBreakingThreshold);
				}
				else
				{
					doOverlapTest = true;   
				}
			}
		
			if(initialContacts == 0 || bLostContacts || doOverlapTest)
			{
				return fullContactsGenerationCapsuleConvex(capsule, convexHull, aToB, transf0, transf1, manifoldContacts, contactBuffer, idtScale, manifold, output.normal, 
					output.closestB, convexHull.getMarginF(), contactDist, doOverlapTest, renderOutput, toleranceLength);
			}
			else
			{
				//This contact is either come from GJK or EPA
				const Vec3V normal = transf1.rotate(output.normal);
				manifold.addManifoldContactsToContactBuffer(contactBuffer, normal, normal, transf0, capsuleRadius, contactDist);
				return true;
			}
		}	
	}
	else if (manifold.getNumContacts() > 0)
	{
		const Vec3V normal = manifold.getWorldNormal(transf1);
		manifold.addManifoldContactsToContactBuffer(contactBuffer, normal, normal, transf0, capsuleRadius, contactDist);
		return true;
	}
	return false;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactPlaneBox.cpp
using namespace px;

EHDI bool Gu::pcmContactPlaneBox(GU_CONTACT_METHOD_ARGS)
{
	E_UNUSED(shape0);
	E_UNUSED(renderOutput);

	using namespace aos;

	PersistentContactManifold& manifold = cache.getManifold();
	PxPrefetchLine(&manifold, 256);

	// Get actual shape data
	const PxBoxGeometry& shapeBox = checkedCast<PxBoxGeometry>(shape1);

	const PxTransformV transf0 = loadTransformA(transform1);//box transform
	const PxTransformV transf1 = loadTransformA(transform0);//plane transform

	//box to plane
	const PxTransformV curTransf(transf1.transformInv(transf0));

	//in world space
	const Vec3V negPlaneNormal = V3Normalize(V3Neg(QuatGetBasisVector0(transf1.q)));
	
	const FloatV contactDist = FLoad(params.mContactDistance);

	const Vec3V boxExtents = V3LoadU(shapeBox.halfExtents);

	const PxReal toleranceLength = params.mToleranceLength;
	const FloatV boxMargin = CalculatePCMBoxMargin(boxExtents, toleranceLength);
	const FloatV projectBreakingThreshold = FMul(boxMargin, FLoad(0.2f));
	const PxU32 initialContacts = manifold.mNumContacts;
	
	manifold.refreshContactPoints(curTransf, projectBreakingThreshold, contactDist);

	const PxU32 newContacts = manifold.mNumContacts;
	const bool bLostContacts = (newContacts != initialContacts);//((initialContacts == 0) || (newContacts != initialContacts));

	if(bLostContacts || manifold.invalidate_PrimitivesPlane(curTransf, boxMargin, FLoad(0.2f)))
	{
		//ML:localNormal is the local space of plane normal, however, because shape1 is box and shape0 is plane, we need to use the reverse of contact normal(which will be the plane normal) to make the refreshContactPoints
		//work out the correct pentration for points
		const Vec3V localNormal = V3UnitX();

		manifold.mNumContacts = 0;
		manifold.setRelativeTransform(curTransf);

		const PxMatTransformV aToB(curTransf);
		const FloatV bx = V3GetX(boxExtents);
		const FloatV by = V3GetY(boxExtents);
		const FloatV bz = V3GetZ(boxExtents);

		const FloatV nbx = FNeg(bx);
		const FloatV nby = FNeg(by);
		const FloatV nbz = FNeg(bz);

		const Vec3V temp0 = V3Scale(aToB.getCol0(), bx);
		const Vec3V temp1 = V3Scale(aToB.getCol1(), by);
		const Vec3V temp2 = V3Scale(aToB.getCol2(), bz);

		const Vec3V ntemp2 = V3Neg(temp2);

		const FloatV px = V3GetX(aToB.p);

		//box's points in the local space of plane
		const Vec3V temp01 = V3Add(temp0, temp1);//(x, y)
		const Vec3V temp02 = V3Sub(temp0, temp1);//(x, -y)

		const FloatV s0 = V3GetX(V3Add(temp2,	temp01));//(x, y, z)
		const FloatV s1 = V3GetX(V3Add(ntemp2,	temp01));//(x, y, -z)
		const FloatV s2 = V3GetX(V3Add(temp2,	temp02));//(x, -y, z)
		const FloatV s3 = V3GetX(V3Add(ntemp2,	temp02));//(x, -y, -z)
		const FloatV s4 = V3GetX(V3Sub(temp2,	temp02));//(-x, y, z)
		const FloatV s5 = V3GetX(V3Sub(ntemp2,	temp02));//(-x, y, -z)
		const FloatV s6 = V3GetX(V3Sub(temp2,	temp01));//(-x, -y, z)
		const FloatV s7 = V3GetX(V3Sub(ntemp2,	temp01));//(-x, -y, -z)

		const FloatV acceptanceDist = FSub(contactDist, px);

		PersistentContact* manifoldContacts = EPX_CP_TO_PCP(contactBuffer.contacts);
		PxU32 numContacts = 0;

		if(FAllGrtr(acceptanceDist, s0))
		{
			const FloatV pen = FAdd(s0, px);
			//(x, y, z)
			manifoldContacts[numContacts].mLocalPointA = boxExtents;//aToB.transformInv(p); 
			manifoldContacts[numContacts].mLocalPointB = V3NegScaleSub(localNormal, pen, aToB.transform(boxExtents)); 
			manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
		}
	
		if(FAllGrtr(acceptanceDist, s1))
		{
			const FloatV pen = FAdd(s1, px);
			//(x, y, -z)
			const Vec3V p = V3Merge(bx, by, nbz);
			//add to contact stream
			manifoldContacts[numContacts].mLocalPointA = p;//aToB.transformInv(p); 
			manifoldContacts[numContacts].mLocalPointB = V3NegScaleSub(localNormal, pen, aToB.transform(p)); 
			manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
		}

		if(FAllGrtr(acceptanceDist, s2))
		{
			const FloatV pen = FAdd(s2, px);
			//(x, -y, z)
			const Vec3V p = V3Merge(bx, nby, bz);
			manifoldContacts[numContacts].mLocalPointA = p;//aToB.transformInv(p); 
			manifoldContacts[numContacts].mLocalPointB = V3NegScaleSub(localNormal, pen, aToB.transform(p)); 
			manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
		}

		if(FAllGrtr(acceptanceDist, s3))
		{
			const FloatV pen = FAdd(s3, px);
			//(x, -y, -z)
			const Vec3V p = V3Merge(bx, nby, nbz);
			manifoldContacts[numContacts].mLocalPointA = p; 
			manifoldContacts[numContacts].mLocalPointB = V3NegScaleSub(localNormal, pen, aToB.transform(p)); 
			manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
		}
	
		if(FAllGrtr(acceptanceDist, s4))
		{
			const FloatV pen = FAdd(s4, px);
			//(-x, y, z)
			const Vec3V p = V3Merge(nbx, by, bz);
			manifoldContacts[numContacts].mLocalPointA = p;
			manifoldContacts[numContacts].mLocalPointB = V3NegScaleSub(localNormal, pen, aToB.transform(p)); 
			manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
		}

		if(FAllGrtr(acceptanceDist, s5))
		{
			const FloatV pen = FAdd(s5, px);
			//(-x, y, -z)
			const Vec3V p = V3Merge(nbx, by, nbz);
			manifoldContacts[numContacts].mLocalPointA = p;//aToB.transformInv(p); 
			manifoldContacts[numContacts].mLocalPointB = V3NegScaleSub(localNormal, pen, aToB.transform(p)); 
			manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
		}
	
		if(FAllGrtr(acceptanceDist, s6))
		{
			const FloatV pen = FAdd(s6, px);
			//(-x, -y, z)
			const Vec3V p = V3Merge(nbx, nby, bz);
			manifoldContacts[numContacts].mLocalPointA = p;//aToB.transformInv(p); 
			manifoldContacts[numContacts].mLocalPointB = V3NegScaleSub(localNormal, pen, aToB.transform(p)); 
			manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
		}
	
		if(FAllGrtr(acceptanceDist, s7))
		{
			const FloatV pen = FAdd(s7, px);
			//(-x, -y, -z)
			const Vec3V p = V3Merge(nbx, nby, nbz);
			manifoldContacts[numContacts].mLocalPointA = p;
			manifoldContacts[numContacts].mLocalPointB = V3NegScaleSub(localNormal, pen, aToB.transform(p)); 
			manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), pen);
		}
		
		//reduce contacts
		manifold.addBatchManifoldContactsCluster(manifoldContacts, numContacts);
		manifold.addManifoldContactsToContactBuffer(contactBuffer, negPlaneNormal, transf1, contactDist);

		return manifold.getNumContacts() > 0;
	}
	else
	{
		manifold.addManifoldContactsToContactBuffer(contactBuffer, negPlaneNormal, transf1, contactDist);
		
		//manifold.drawManifold(*gRenderOutPut, transf0, transf1);
		return manifold.getNumContacts() > 0;
	}	
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactPlaneCapsule.cpp
using namespace px;

EHDI bool Gu::pcmContactPlaneCapsule(GU_CONTACT_METHOD_ARGS)
{
	E_UNUSED(shape0);
	E_UNUSED(renderOutput);

	using namespace aos;

	PersistentContactManifold& manifold = cache.getManifold();
	PxPrefetchLine(&manifold, 256);

	// Get actual shape data
	const PxCapsuleGeometry& shapeCapsule = checkedCast<PxCapsuleGeometry>(shape1);

	const PxTransformV transf0 = loadTransformA(transform1);//capsule transform
	const PxTransformV transf1 = loadTransformA(transform0);//plane transform
	//capsule to plane
	const PxTransformV aToB(transf1.transformInv(transf0));

	//in world space
	const Vec3V planeNormal = V3Normalize(QuatGetBasisVector0(transf1.q));
	const Vec3V contactNormal = V3Neg(planeNormal);
	
	//ML:localNormal is the local space of plane normal, however, because shape1 is capulse and shape0 is plane, we need to use the reverse of contact normal(which will be the plane normal) to make the refreshContactPoints
	//work out the correct pentration for points
	const Vec3V localNormal = V3UnitX();

	const FloatV contactDist = FLoad(params.mContactDistance);

	const FloatV radius = FLoad(shapeCapsule.radius);
	const FloatV halfHeight = FLoad(shapeCapsule.halfHeight);

	//capsule is in the local space of plane(n = (1.f, 0.f, 0.f), d=0.f)
	const Vec3V basisVector = QuatGetBasisVector0(aToB.q);
	const Vec3V tmp = V3Scale(basisVector, halfHeight);
	const Vec3V s = V3Add(aToB.p, tmp);
	const Vec3V e = V3Sub(aToB.p, tmp);

	const FloatV inflatedRadius = FAdd(radius, contactDist);
	const FloatV replaceBreakingThreshold = FMul(radius, FLoad(0.001f));
	const FloatV projectBreakingThreshold = FMul(radius, FLoad(0.05f));
	const PxU32 initialContacts = manifold.mNumContacts;

	//manifold.refreshContactPoints(curRTrans, projectBreakingThreshold, contactDist);
	const FloatV refreshDist = FAdd(contactDist, radius);
	manifold.refreshContactPoints(aToB, projectBreakingThreshold, refreshDist);

	const PxU32 newContacts = manifold.mNumContacts;
	const bool bLostContacts = (newContacts != initialContacts);//((initialContacts == 0) || (newContacts != initialContacts));

	if(bLostContacts || manifold.invalidate_PrimitivesPlane(aToB, radius, FLoad(0.02f)))  
	{
		manifold.mNumContacts = 0;
		manifold.setRelativeTransform(aToB);
		//calculate the distance from s to the plane
		const FloatV signDist0 = V3GetX(s);//V3Dot(localNormal, s);
		if(FAllGrtr(inflatedRadius, signDist0))
		{
			const Vec3V localPointA = aToB.transformInv(s);
			const Vec3V localPointB = V3NegScaleSub(localNormal, signDist0, s);
			const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), signDist0);
			//add to manifold
			manifold.addManifoldPoint2(localPointA, localPointB, localNormalPen, replaceBreakingThreshold); 
		}

		const FloatV signDist1 = V3GetX(e);//V3Dot(localNormal, e);
		if(FAllGrtr(inflatedRadius, signDist1))
		{
			const Vec3V localPointA = aToB.transformInv(e);
			const Vec3V localPointB = V3NegScaleSub(localNormal, signDist1, e);
			const Vec4V localNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), signDist1);
			//add to manifold
			manifold.addManifoldPoint2(localPointA, localPointB, localNormalPen, replaceBreakingThreshold); 
		}

		manifold.addManifoldContactsToContactBuffer(contactBuffer, contactNormal, planeNormal, transf0, radius, contactDist);


		return manifold.getNumContacts() > 0;
	}
	else
	{
		manifold.addManifoldContactsToContactBuffer(contactBuffer, contactNormal, planeNormal, transf0, radius, contactDist);
		return manifold.getNumContacts() > 0;
	}
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactPlaneConvex.cpp
using namespace px;

EHDI bool Gu::pcmContactPlaneConvex(GU_CONTACT_METHOD_ARGS)
{
	E_UNUSED(shape0);
	E_UNUSED(renderOutput);

	using namespace aos;

	PersistentContactManifold& manifold = cache.getManifold();
	PxPrefetchLine(&manifold, 256);

	// Get actual shape data
	const PxConvexMeshGeometry& shapeConvex = checkedCast<PxConvexMeshGeometry>(shape1);

	const PxTransformV transf0 = loadTransformA(transform1);//convex transform
	const PxTransformV transf1 = loadTransformA(transform0);//plane transform
	//convex to plane
	const PxTransformV curTransf(transf1.transformInv(transf0));
	
	const Vec3V vScale = V3LoadU_SafeReadW(shapeConvex.scale.scale);	// PT: safe because 'rotation' follows 'scale' in PxMeshScale
	const ConvexHullData* hullData = _getHullData(shapeConvex);

	const PxReal toleranceLength = params.mToleranceLength;
	const FloatV convexMargin = CalculatePCMConvexMargin(hullData, vScale, toleranceLength);
	
	//in world space
	const Vec3V planeNormal = V3Normalize(QuatGetBasisVector0(transf1.q));
	const Vec3V negPlaneNormal = V3Neg(planeNormal);
	
	const FloatV contactDist = FLoad(params.mContactDistance);

	//const FloatV replaceBreakingThreshold = FMul(convexMargin, FLoad(0.001f));
	const FloatV projectBreakingThreshold = FMul(convexMargin, FLoad(0.2f));
	const PxU32 initialContacts = manifold.mNumContacts;
	
	manifold.refreshContactPoints(curTransf, projectBreakingThreshold, contactDist);

	const PxU32 newContacts = manifold.mNumContacts;
	const bool bLostContacts = (newContacts != initialContacts);//((initialContacts == 0) || (newContacts != initialContacts));

	if(bLostContacts || manifold.invalidate_PrimitivesPlane(curTransf, convexMargin, FLoad(0.2f)))
	{
		const PxMatTransformV aToB(curTransf);
		const QuatV vQuat = QuatVLoadU(&shapeConvex.scale.rotation.x);

		const Mat33V vertex2Shape = ConstructVertex2ShapeMatrix(vScale, vQuat);
		
		//ML:localNormal is the local space of plane normal, however, because shape1 is box and shape0 is plane, we need to use the reverse of contact normal(which will be the plane normal) to make the refreshContactPoints
		//work out the correct pentration for points
		const Vec3V localNormal = V3UnitX();

		manifold.mNumContacts = 0;
		manifold.setRelativeTransform(curTransf);
		const PxVec3*  verts = hullData->getHullVertices();

		const PxU32 nbPolygons = hullData->mNbPolygons;

		const Vec3V n = V3Normalize(M33MulV3(vertex2Shape, aToB.rotateInv(localNormal)));
		const Vec3V nnormal = V3Neg(n);
		const FloatV zero = FZero();

		PersistentContact* manifoldContacts = EPX_CP_TO_PCP(contactBuffer.contacts);
		PxU32 numContacts = 0;

		const PxMatTransformV aToBVertexSpace(aToB.p, M33MulM33(aToB.rot, vertex2Shape));

		FloatV minProj = FMax();
		PxU32 closestFaceIndex = 0;
		PxU32 polyIndex2 = 0xFFFFFFFF;

		for (PxU32 i = 0; i < nbPolygons; ++i)
		{
			const HullPolygonData& polyData = hullData->mPolygons[i];
			const Vec3V planeN = V3LoadU_SafeReadW(polyData.mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class
			const FloatV proj = V3Dot(n, planeN);
			if (FAllGrtr(minProj, proj))
			{
				minProj = proj;
				closestFaceIndex = PxI32(i);
			}
		}

		const PxU32 numEdges = hullData->mNbEdges;
		const PxU8* const edgeToFace = hullData->getFacesByEdges8();

		//Loop through edges
		PxU32 closestEdge = 0xffffffff;
		//We subtract a small bias to increase the chances of selecting an edge below
		minProj = FSub(minProj, FLoad(5e-4f));
		FloatV maxDpSq = FMul(minProj, minProj);

		for (PxU32 i = 0; i < numEdges; ++i)//, inc = VecI32V_Add(inc, vOne))
		{
			const PxU32 index = i * 2;
			const PxU8 f0 = edgeToFace[index];
			const PxU8 f1 = edgeToFace[index + 1];

			const Vec3V planeNormal0 = V3LoadU_SafeReadW(hullData->mPolygons[f0].mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class
			const Vec3V planeNormal1 = V3LoadU_SafeReadW(hullData->mPolygons[f1].mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class

			// unnormalized edge normal
			const Vec3V edgeNormal = V3Add(planeNormal0, planeNormal1);//polys[f0].mPlane.n + polys[f1].mPlane.n;
			const FloatV enMagSq = V3Dot(edgeNormal, edgeNormal);//edgeNormal.magnitudeSquared();
			//Test normal of current edge - squared test is valid if dp and maxDp both >= 0
			const FloatV dp = V3Dot(edgeNormal, nnormal);//edgeNormal.dot(normal);
			const FloatV sqDp = FMul(dp, dp);

			const BoolV con0 = FIsGrtrOrEq(dp, zero);
			const BoolV con1 = FIsGrtr(sqDp, FMul(maxDpSq, enMagSq));
			const BoolV con = BAnd(con0, con1);
			if (BAllEqTTTT(con))
			{
				maxDpSq = FDiv(sqDp, enMagSq);
				closestEdge = i;
			}
		}

		if (closestEdge != 0xffffffff)
		{
			const PxU8* FBE = edgeToFace;

			const PxU32 index = closestEdge * 2;
			const PxU32 f0 = FBE[index];
			const PxU32 f1 = FBE[index + 1];

			const Vec3V planeNormal0 = V3LoadU_SafeReadW(hullData->mPolygons[f0].mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class
			const Vec3V planeNormal1 = V3LoadU_SafeReadW(hullData->mPolygons[f1].mPlane.n);	// PT: safe because 'd' follows 'n' in the plane class

			const FloatV dp0 = V3Dot(planeNormal0, nnormal);
			const FloatV dp1 = V3Dot(planeNormal1, nnormal);
			if (FAllGrtr(dp0, dp1))
			{
				closestFaceIndex = PxI32(f0);
				polyIndex2 = PxI32(f1);
			}
			else
			{
				closestFaceIndex = PxI32(f1);
				polyIndex2 = PxI32(f0);
			}
		}

		for (PxU32 index = closestFaceIndex; index != 0xFFFFFFFF; index = polyIndex2, polyIndex2 = 0xFFFFFFFF)
		{
			const HullPolygonData& face = hullData->mPolygons[closestFaceIndex];
			const PxU32 nbFaceVerts = face.mNbVerts;

			const PxU8* vertInds = hullData->getVertexData8() + face.mVRef8;

			for (PxU32 i = 0; i < nbFaceVerts; ++i)
			{
				const Vec3V pInVertexSpace = V3LoadU(verts[vertInds[i]]);

				//transform p into plane space
				const Vec3V pInPlaneSpace = aToBVertexSpace.transform(pInVertexSpace);//V3Add(aToB.p, M33MulV3(temp1, pInVertexSpace));

				const FloatV signDist = V3GetX(pInPlaneSpace);

				if (FAllGrtr(contactDist, signDist))
				{
					//transform p into shape space
					const Vec3V pInShapeSpace = M33MulV3(vertex2Shape, pInVertexSpace);
					//add to manifold

					manifoldContacts[numContacts].mLocalPointA = pInShapeSpace;
					manifoldContacts[numContacts].mLocalPointB = V3NegScaleSub(localNormal, signDist, pInPlaneSpace);
					manifoldContacts[numContacts++].mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(localNormal), signDist);

					if (numContacts == 64)
					{
						manifold.reduceBatchContactsCluster(manifoldContacts, numContacts);
						numContacts = GU_MANIFOLD_CACHE_SIZE;

						for (PxU32 c = 0; c < GU_MANIFOLD_CACHE_SIZE; ++c)
							manifoldContacts[c] = manifold.mContactPoints[c];
					}
				}
			}
		}

		//reduce contacts
		//manifold.addBatchManifoldContactsCluster(manifoldContacts, numContacts);

		manifold.addBatchManifoldContacts(manifoldContacts, numContacts, toleranceLength);
	}
	
	manifold.addManifoldContactsToContactBuffer(contactBuffer, negPlaneNormal, transf1, contactDist);
	return manifold.getNumContacts() > 0;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactSphereBox.cpp
using namespace px;

EHDI bool Gu::pcmContactSphereBox(GU_CONTACT_METHOD_ARGS)
{
	E_UNUSED(renderOutput);
	E_UNUSED(cache);

	using namespace aos;
	// Get actual shape data
	const PxSphereGeometry& shapeSphere = checkedCast<PxSphereGeometry>(shape0);
	const PxBoxGeometry& shapeBox = checkedCast<PxBoxGeometry>(shape1);

	//const PsTransformV transf0(transform0);
	const Vec3V sphereOrigin = V3LoadA(&transform0.p.x);
	//const PsTransformV transf1(transform1);

	const QuatV q1 = QuatVLoadA(&transform1.q.x);
	const Vec3V p1 = V3LoadA(&transform1.p.x);

	const FloatV radius = FLoad(shapeSphere.radius);
	
	const PxTransformV transf1(p1, q1);
	
	const FloatV cDist = FLoad(params.mContactDistance);

	const Vec3V boxExtents = V3LoadU(shapeBox.halfExtents);

	//translate sphere center into the box space
	const Vec3V sphereCenter = transf1.transformInv(sphereOrigin);

	const Vec3V nBoxExtents = V3Neg(boxExtents);

	//const FloatV radSq = FMul(radius, radius);

	const FloatV inflatedSum = FAdd(radius, cDist);
	const FloatV sqInflatedSum = FMul(inflatedSum, inflatedSum);

	const Vec3V p = V3Clamp(sphereCenter, nBoxExtents, boxExtents);
	const Vec3V v = V3Sub(sphereCenter, p);
	const FloatV lengthSq = V3Dot(v, v);

	E_ASSERT(contactBuffer.count < PxContactBuffer::MAX_CONTACTS);

	if(FAllGrtr(sqInflatedSum, lengthSq))//intersect
	{
		//check whether the spherCenter is inside the box
		const BoolV bInsideBox = V3IsGrtrOrEq(boxExtents, V3Abs(sphereCenter));
		// PT: TODO: ??? revisit this, why do we have both BAllEqTTTT and BAllTrue3?
		if(BAllEqTTTT(BAllTrue3(bInsideBox)))//sphere center inside the box
		{
			//Pick directions and sign
			const Vec3V absP = V3Abs(p);
			const Vec3V distToSurface = V3Sub(boxExtents, absP);//dist from embedded center to box surface along 3 dimensions.
			
			const FloatV x = V3GetX(distToSurface);
			const FloatV y = V3GetY(distToSurface);
			const FloatV z = V3GetZ(distToSurface);

			const Vec3V xV = V3Splat(x);
			const Vec3V zV = V3Splat(z);

			//find smallest element of distToSurface
			const BoolV con0 = BAllTrue3(V3IsGrtrOrEq(distToSurface, zV));
			const BoolV con1 = BAllTrue3(V3IsGrtrOrEq(distToSurface, xV));
			const Vec3V sign = V3Sign(p);
	
			const Vec3V tmpX = V3Mul(V3UnitX(), sign);
			const Vec3V tmpY = V3Mul(V3UnitY(), sign);
			const Vec3V tmpZ = V3Mul(V3UnitZ(), sign);
			
			const Vec3V locNorm = V3Sel(con0, tmpZ, V3Sel(con1, tmpX, tmpY));////local coords contact normal
			const FloatV dist = FNeg(FSel(con0, z, FSel(con1, x, y)));

			//separation so far is just the embedding of the center point; we still have to push out all of the radius.
			const Vec3V normal = transf1.rotate(locNorm);
			const FloatV penetration = FSub(dist, radius);
			const Vec3V point = V3Sub(sphereOrigin , V3Scale(normal, dist));

			outputSimplePCMContact(contactBuffer, point, normal, penetration);
		}
		else
		{
			//get the closest point from the center to the box surface
			const FloatV recipLength = FRsqrt(lengthSq);
			const FloatV length = FRecip(recipLength);
			const Vec3V locNorm = V3Scale(v, recipLength);
			const FloatV penetration = FSub(length, radius);
			const Vec3V normal = transf1.rotate(locNorm);
			const Vec3V point = transf1.transform(p);

			outputSimplePCMContact(contactBuffer, point, normal, penetration);
		}
		return true;
	}
	return false;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactSphereCapsule.cpp
using namespace px;
using namespace aos;

static EHD FloatV distancePointSegmentSquared(const Vec3VArg a, const Vec3VArg b, const Vec3VArg p, FloatV& param)
{
	const FloatV zero = FZero();
	const FloatV one = FOne();

	const Vec3V ap = V3Sub(p, a);
	const Vec3V ab = V3Sub(b, a);
	const FloatV nom = V3Dot(ap, ab);
	
	const FloatV denom = V3Dot(ab, ab);
	const FloatV tValue = FClamp(FDiv(nom, denom), zero, one);

	const FloatV t = FSel(FIsEq(denom, zero), zero, tValue);
	const Vec3V v = V3NegScaleSub(ab, t, ap);
	param = t;
	return V3Dot(v, v);
}

EHDI bool Gu::pcmContactSphereCapsule(GU_CONTACT_METHOD_ARGS)
{
	E_UNUSED(cache);
	E_UNUSED(renderOutput);

	const PxSphereGeometry& shapeSphere = checkedCast<PxSphereGeometry>(shape0);
	const PxCapsuleGeometry& shapeCapsule = checkedCast<PxCapsuleGeometry>(shape1);

	//Sphere in world space
	const Vec3V sphereCenter = V3LoadA(&transform0.p.x);
	const QuatV q1 = QuatVLoadA(&transform1.q.x);
	const Vec3V p1 = V3LoadA(&transform1.p.x);

	const FloatV sphereRadius = FLoad(shapeSphere.radius);
	const FloatV capsuleRadius = FLoad(shapeCapsule.radius);
	const FloatV cDist = FLoad(params.mContactDistance);

	//const FloatV r0 = FloatV_From_F32(shapeCapsule.radius);
	const FloatV halfHeight0 = FLoad(shapeCapsule.halfHeight);
	const Vec3V basisVector0 = QuatGetBasisVector0(q1);
	const Vec3V tmp0 = V3Scale(basisVector0, halfHeight0);
	const Vec3V s = V3Add(p1, tmp0);
	const Vec3V e = V3Sub(p1, tmp0);

	const FloatV radiusSum = FAdd(sphereRadius, capsuleRadius);
	const FloatV inflatedSum = FAdd(radiusSum, cDist);

	// Collision detection
	FloatV t;
	const FloatV squareDist = distancePointSegmentSquared(s, e, sphereCenter, t);
	const FloatV sqInflatedSum = FMul(inflatedSum, inflatedSum);

	if(FAllGrtr(sqInflatedSum, squareDist))//BAllEq(con, bTrue))
	{
		const Vec3V p = V3ScaleAdd(V3Sub(e, s), t, s);
		const Vec3V dir = V3Sub(sphereCenter, p);
		const Vec3V normal = V3NormalizeSafe(dir, V3UnitX());
		const Vec3V point = V3NegScaleSub(normal, sphereRadius, sphereCenter);//transform back to the world space

		const FloatV dist = FSub(FSqrt(squareDist), radiusSum);
		//context.mContactBuffer.contact(point, normal, FSub(FSqrt(squareDist), radiusSum));

		return outputSimplePCMContact(contactBuffer, point, normal, dist);
	}
	return false;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactSphereConvex.cpp
using namespace px;
using namespace Gu;
using namespace aos;

EHDI static void addToContactBuffer(PxContactBuffer& contactBuffer, const Vec3VArg worldNormal, const Vec3VArg worldPoint, const FloatVArg penDep)
{
	outputSimplePCMContact(contactBuffer, worldPoint, worldNormal, penDep);
}

EHDI static bool fullContactsGenerationSphereConvex(const CapsuleV& capsule, const ConvexHullV& convexHull, const PxTransformV& transf0, const PxTransformV& transf1,
								PersistentContact* manifoldContacts, PxContactBuffer& contactBuffer, const bool idtScale, PersistentContactManifold& manifold, 
								Vec3VArg normal, const FloatVArg contactDist, bool doOverlapTest, PxRenderOutput* renderOutput)
{
	E_UNUSED(renderOutput);

	PolygonalData polyData;
	getPCMConvexData(convexHull,idtScale, polyData);

	alignas(16) PxU8 buff[sizeof(SupportLocalImpl<ConvexHullV>)];
	SupportLocal* map = (idtScale ? static_cast<SupportLocal*>(E_PLACEMENT_NEW(buff, SupportLocalImpl<ConvexHullNoScaleV>)(static_cast<const ConvexHullNoScaleV&>(convexHull), transf1, convexHull.vertex2Shape, convexHull.shape2Vertex, idtScale)) : 
	static_cast<SupportLocal*>(E_PLACEMENT_NEW(buff, SupportLocalImpl<ConvexHullV>)(convexHull, transf1, convexHull.vertex2Shape, convexHull.shape2Vertex, idtScale)));

	PxU32 numContacts = 0;
	if(generateSphereFullContactManifold(capsule, polyData, map, manifoldContacts, numContacts, contactDist, normal, doOverlapTest))
	{
		if(numContacts > 0)
		{
			PersistentContact& p = manifold.getContactPoint(0);

			p.mLocalPointA = manifoldContacts[0].mLocalPointA;
			p.mLocalPointB = manifoldContacts[0].mLocalPointB;
			p.mLocalNormalPen = manifoldContacts[0].mLocalNormalPen;
			manifold.mNumContacts =1;

			//transform normal to world space
			const Vec3V worldNormal = transf1.rotate(normal);
			const Vec3V worldP = V3NegScaleSub(worldNormal, capsule.radius, transf0.p);
			const FloatV penDep = FSub(V4GetW(manifoldContacts[0].mLocalNormalPen), capsule.radius);

			addToContactBuffer(contactBuffer, worldNormal, worldP, penDep);

			return true;
		}
	}

	return false;
}

EHDI bool Gu::pcmContactSphereConvex(GU_CONTACT_METHOD_ARGS)
{
	E_ASSERT(transform1.q.isSane());
	E_ASSERT(transform0.q.isSane());
	
	const PxConvexMeshGeometry& shapeConvex = checkedCast<PxConvexMeshGeometry>(shape1);
	const PxSphereGeometry& shapeSphere = checkedCast<PxSphereGeometry>(shape0);

	PersistentContactManifold& manifold = cache.getManifold();

	const Vec3V zeroV = V3Zero();

	const ConvexHullData* hullData = _getHullData(shapeConvex);
	PxPrefetchLine(hullData);
	const Vec3V vScale = V3LoadU_SafeReadW(shapeConvex.scale.scale);	// PT: safe because 'rotation' follows 'scale' in PxMeshScale
	const FloatV sphereRadius = FLoad(shapeSphere.radius);
	const FloatV contactDist = FLoad(params.mContactDistance);
	
	//Transfer A into the local space of B
	const PxTransformV transf0 = loadTransformA(transform0);
	const PxTransformV transf1 = loadTransformA(transform1);
	const PxTransformV curRTrans(transf1.transformInv(transf0));
	const PxMatTransformV aToB(curRTrans);
	
	const PxReal toleranceLength = params.mToleranceLength;
	const FloatV convexMargin = CalculatePCMConvexMargin(hullData, vScale, toleranceLength);

	const PxU32 initialContacts = manifold.mNumContacts;
	const FloatV minMargin = FMin(convexMargin, sphereRadius);
	const FloatV projectBreakingThreshold = FMul(minMargin, FLoad(0.05f));
	
	const FloatV refreshDistance = FAdd(sphereRadius, contactDist);
	manifold.refreshContactPoints(aToB, projectBreakingThreshold, refreshDistance);
	//ML: after refreshContactPoints, we might lose some contacts
	const bool bLostContacts = (manifold.mNumContacts != initialContacts);

	if(bLostContacts || manifold.invalidate_SphereCapsule(curRTrans, minMargin))
	{
		GjkStatus status = manifold.mNumContacts > 0 ? GJK_UNDEFINED : GJK_NON_INTERSECT;

		manifold.setRelativeTransform(curRTrans);
		
		const QuatV vQuat = QuatVLoadU(&shapeConvex.scale.rotation.x);  
	
		const bool idtScale = shapeConvex.scale.isIdentity();
		//use the original shape
		const ConvexHullV convexHull(hullData, V3LoadU(hullData->mCenterOfMass), vScale, vQuat, idtScale);
		//transform capsule into the local space of convexHull
		const CapsuleV capsule(aToB.p, sphereRadius);

		GjkOutput output;
		const LocalConvex<CapsuleV> convexA(capsule);
		const Vec3V initialSearchDir = V3Sub(capsule.getCenter(), convexHull.getCenter());
		if(idtScale)
		{
			const LocalConvex<ConvexHullNoScaleV> convexB(*EPX_CONVEX_TO_NOSCALECONVEX(&convexHull));
			status = gjkPenetration<LocalConvex<CapsuleV>, LocalConvex<ConvexHullNoScaleV> >(convexA, convexB, initialSearchDir, contactDist, true,
				manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);
		}
		else
		{
			const LocalConvex<ConvexHullV> convexB(convexHull);
			status = gjkPenetration<LocalConvex<CapsuleV>, LocalConvex<ConvexHullV> >(convexA, convexB, initialSearchDir, contactDist, true,
				manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints, output);
		}

		if(status == GJK_NON_INTERSECT)
		{
			return false;
		}
		else if(status == GJK_CONTACT)
		{
			PersistentContact& p = manifold.getContactPoint(0);
			p.mLocalPointA = zeroV;//sphere center
			p.mLocalPointB = output.closestB;
			p.mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(output.normal), output.penDep);
			manifold.mNumContacts =1;

			
			//transform normal to world space
			const Vec3V worldNormal = transf1.rotate(output.normal);
			const Vec3V worldP = V3NegScaleSub(worldNormal, sphereRadius, transf0.p);
			const FloatV penDep = FSub(output.penDep, sphereRadius);
			addToContactBuffer(contactBuffer, worldNormal, worldP, penDep);
			return true;
		}
		else if(status == GJK_DEGENERATE)
		{
			PersistentContact* manifoldContacts = EPX_CP_TO_PCP(contactBuffer.contacts);
			
			return fullContactsGenerationSphereConvex(capsule, convexHull, transf0, transf1, manifoldContacts, contactBuffer, idtScale, 
				manifold, output.normal, contactDist, true, renderOutput);
		}
		else if (status == EPA_CONTACT)
		{
			if (idtScale)
			{
				const LocalConvex<ConvexHullNoScaleV> convexB(*EPX_CONVEX_TO_NOSCALECONVEX(&convexHull));
				status = epaPenetration(convexA, convexB, manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints,
						true, FLoad(toleranceLength), output);
			}
			else
			{
				const LocalConvex<ConvexHullV> convexB(convexHull);			
				status = epaPenetration(convexA, convexB, manifold.mAIndice, manifold.mBIndice, manifold.mNumWarmStartPoints,
						true, FLoad(toleranceLength), output);
			}

			if (status == EPA_CONTACT)
			{
				PersistentContact& p = manifold.getContactPoint(0);
				p.mLocalPointA = zeroV;//sphere center
				p.mLocalPointB = output.closestB;
				p.mLocalNormalPen = V4SetW(Vec4V_From_Vec3V(output.normal), output.penDep);
				manifold.mNumContacts = 1;

				//transform normal to world space
				const Vec3V worldNormal = transf1.rotate(output.normal);
				const Vec3V worldP = V3NegScaleSub(worldNormal, sphereRadius, transf0.p);
				const FloatV penDep = FSub(output.penDep, sphereRadius);

				addToContactBuffer(contactBuffer, worldNormal, worldP, penDep);
				return true;
			}
			else
			{
				PersistentContact* manifoldContacts = EPX_CP_TO_PCP(contactBuffer.contacts);
				return fullContactsGenerationSphereConvex(capsule, convexHull, transf0, transf1, manifoldContacts, contactBuffer, idtScale,
					manifold, output.normal, contactDist, true, renderOutput);
			}
		}
	}
	else if(manifold.mNumContacts > 0)
	{
		//ML:: the manifold originally has contacts
		PersistentContact& p = manifold.getContactPoint(0);
		const Vec3V worldNormal = transf1.rotate(Vec3V_From_Vec4V(p.mLocalNormalPen));
		const Vec3V worldP = V3NegScaleSub(worldNormal, sphereRadius, transf0.p);
		const FloatV penDep = FSub(V4GetW(p.mLocalNormalPen), sphereRadius);

		addToContactBuffer(contactBuffer, worldNormal, worldP, penDep);
		return true;
	}

	return false;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactSpherePlane.cpp
using namespace px;

EHDI bool Gu::pcmContactSpherePlane(GU_CONTACT_METHOD_ARGS)
{
	using namespace aos;
	E_UNUSED(renderOutput);
	E_UNUSED(cache);
	E_UNUSED(shape1);

	// Get actual shape data
	const PxSphereGeometry& shapeSphere = checkedCast<PxSphereGeometry>(shape0);

	//sphere transform
	const Vec3V p0 = V3LoadU_SafeReadW(transform0.p);	// PT: safe because 'mRefCount' follows 'mTransform' in PxsTransform

	//plane transform
	const Vec3V p1 = V3LoadU_SafeReadW(transform1.p);	// PT: safe because 'mRefCount' follows 'mTransform' in PxsTransform
	const QuatV q1 = QuatVLoadU(&transform1.q.x);

	const FloatV radius = FLoad(shapeSphere.radius);
	const FloatV contactDist = FLoad(params.mContactDistance);

	const PxTransformV transf1(p1, q1);
	//Sphere in plane space
	const Vec3V sphereCenterInPlaneSpace = transf1.transformInv(p0);
	
	//Separation
	const FloatV separation = FSub(V3GetX(sphereCenterInPlaneSpace), radius);

	if(FAllGrtrOrEq(contactDist, separation))
	{
		//get the plane normal
		const Vec3V worldNormal = QuatGetBasisVector0(q1);
		const Vec3V worldPoint = V3NegScaleSub(worldNormal, radius, p0);

		return outputSimplePCMContact(contactBuffer, worldPoint, worldNormal, separation);
	}
	return false;
}

// ===== 원본: physx/source/geomutils/src/pcm/GuPCMContactSphereSphere.cpp
using namespace px;

EHDI bool Gu::pcmContactSphereSphere(GU_CONTACT_METHOD_ARGS)
{
	E_UNUSED(cache);
	E_UNUSED(renderOutput);

	using namespace aos;
	const PxSphereGeometry& shapeSphere0 = checkedCast<PxSphereGeometry>(shape0);
	const PxSphereGeometry& shapeSphere1 = checkedCast<PxSphereGeometry>(shape1);
	
	const FloatV cDist = FLoad(params.mContactDistance);
	const Vec3V p0 = V3LoadA(&transform0.p.x);
	const Vec3V p1 = V3LoadA(&transform1.p.x);

	const FloatV r0	= FLoad(shapeSphere0.radius);
	const FloatV r1	= FLoad(shapeSphere1.radius);
	
	const Vec3V _delta = V3Sub(p0, p1);
	const FloatV distanceSq = V3Dot(_delta, _delta);
	const FloatV radiusSum = FAdd(r0, r1);
	const FloatV inflatedSum = FAdd(radiusSum, cDist);
	
	if(FAllGrtr(FMul(inflatedSum, inflatedSum), distanceSq))
	{
		const FloatV eps = FLoad(0.00001f);
		const FloatV dist = FSqrt(distanceSq);
		const BoolV bCon = FIsGrtrOrEq(eps, dist);
		const Vec3V normal = V3Sel(bCon, V3UnitX(), V3ScaleInv(_delta, dist));
		const Vec3V point = V3ScaleAdd(normal, r1, p1);
		const FloatV pen = FSub(dist, radiusSum);
		
		return outputSimplePCMContact(contactBuffer, point, normal, pen);
	}
	return false;
}

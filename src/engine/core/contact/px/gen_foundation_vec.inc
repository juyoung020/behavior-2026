// ===== 원본: physx/include/foundation/PxVec2.h
namespace px
{

/**
\brief 2 Element vector class.

This is a 2-dimensional vector class with public data members.
*/
template<class Type>
class PxVec2T
{
  public:
	/**
	\brief default constructor leaves data uninitialized.
	*/
	EHD PxVec2T()
	{
	}

	/**
	\brief zero constructor.
	*/
	EHD PxVec2T(PxZERO) : x(Type(0.0)), y(Type(0.0))
	{
	}

	/**
	\brief Assigns scalar parameter to all elements.

	Useful to initialize to zero or one.

	\param[in] a Value to assign to elements.
	*/
	explicit EHD PxVec2T(Type a) : x(a), y(a)
	{
	}

	/**
	\brief Initializes from 2 scalar parameters.

	\param[in] nx Value to initialize X component.
	\param[in] ny Value to initialize Y component.
	*/
	EHD PxVec2T(Type nx, Type ny) : x(nx), y(ny)
	{
	}

	/**
	\brief Copy ctor.
	*/
	EHD PxVec2T(const PxVec2T& v) : x(v.x), y(v.y)
	{
	}

	// Operators

	/**
	\brief Assignment operator
	*/
	EHD PxVec2T& operator=(const PxVec2T& p)
	{
		x = p.x;
		y = p.y;
		return *this;
	}

	/**
	\brief element access
	*/
	EHD Type& operator[](unsigned int index)
	{
		E_ASSERT(index <= 1);
		return reinterpret_cast<Type*>(this)[index];
	}

	/**
	\brief element access
	*/
	EHD const Type& operator[](unsigned int index) const
	{
		E_ASSERT(index <= 1);
		return reinterpret_cast<const Type*>(this)[index];
	}

	/**
	\brief returns true if the two vectors are exactly equal.
	*/
	EHD bool operator==(const PxVec2T& v) const
	{
		return x == v.x && y == v.y;
	}

	/**
	\brief returns true if the two vectors are not exactly equal.
	*/
	EHD bool operator!=(const PxVec2T& v) const
	{
		return x != v.x || y != v.y;
	}

	/**
	\brief tests for exact zero vector
	*/
	EHD bool isZero() const
	{
		return x == Type(0.0) && y == Type(0.0);
	}

	/**
	\brief returns true if all 2 elems of the vector are finite (not NAN or INF, etc.)
	*/
	EHDI bool isFinite() const
	{
		return PxIsFinite(x) && PxIsFinite(y);
	}

	/**
	\brief is normalized - used by API parameter validation
	*/
	EHD bool isNormalized() const
	{
		const Type unitTolerance = Type(1e-4);
		return isFinite() && PxAbs(magnitude() - Type(1.0)) < unitTolerance;
	}

	/**
	\brief returns the squared magnitude

	Avoids calling PxSqrt()!
	*/
	EHD Type magnitudeSquared() const
	{
		return x * x + y * y;
	}

	/**
	\brief returns the magnitude
	*/
	EHD Type magnitude() const
	{
		return PxSqrt(magnitudeSquared());
	}

	/**
	\brief negation
	*/
	EHD PxVec2T operator-() const
	{
		return PxVec2T(-x, -y);
	}

	/**
	\brief vector addition
	*/
	EHD PxVec2T operator+(const PxVec2T& v) const
	{
		return PxVec2T(x + v.x, y + v.y);
	}

	/**
	\brief vector difference
	*/
	EHD PxVec2T operator-(const PxVec2T& v) const
	{
		return PxVec2T(x - v.x, y - v.y);
	}

	/**
	\brief scalar post-multiplication
	*/
	EHD PxVec2T operator*(Type f) const
	{
		return PxVec2T(x * f, y * f);
	}

	/**
	\brief scalar division
	*/
	EHD PxVec2T operator/(Type f) const
	{
		f = Type(1.0) / f;
		return PxVec2T(x * f, y * f);
	}

	/**
	\brief vector addition
	*/
	EHD PxVec2T& operator+=(const PxVec2T& v)
	{
		x += v.x;
		y += v.y;
		return *this;
	}

	/**
	\brief vector difference
	*/
	EHD PxVec2T& operator-=(const PxVec2T& v)
	{
		x -= v.x;
		y -= v.y;
		return *this;
	}

	/**
	\brief scalar multiplication
	*/
	EHD PxVec2T& operator*=(Type f)
	{
		x *= f;
		y *= f;
		return *this;
	}

	/**
	\brief scalar division
	*/
	EHD PxVec2T& operator/=(Type f)
	{
		f = Type(1.0) / f;
		x *= f;
		y *= f;
		return *this;
	}

	/**
	\brief returns the scalar product of this and other.
	*/
	EHD Type dot(const PxVec2T& v) const
	{
		return x * v.x + y * v.y;
	}

	/** returns a unit vector */
	EHD PxVec2T getNormalized() const
	{
		const Type m = magnitudeSquared();
		return m > Type(0.0) ? *this * PxRecipSqrt(m) : PxVec2T(Type(0));
	}

	/**
	\brief normalizes the vector in place
	*/
	EHD Type normalize()
	{
		const Type m = magnitude();
		if(m > Type(0.0))
			*this /= m;
		return m;
	}

	/**
	\brief a[i] * b[i], for all i.
	*/
	EHD PxVec2T multiply(const PxVec2T& a) const
	{
		return PxVec2T(x * a.x, y * a.y);
	}

	/**
	\brief element-wise minimum
	*/
	EHD PxVec2T minimum(const PxVec2T& v) const
	{
		return PxVec2T(PxMin(x, v.x), PxMin(y, v.y));
	}

	/**
	\brief returns MIN(x, y);
	*/
	EHD Type minElement() const
	{
		return PxMin(x, y);
	}

	/**
	\brief element-wise maximum
	*/
	EHD PxVec2T maximum(const PxVec2T& v) const
	{
		return PxVec2T(PxMax(x, v.x), PxMax(y, v.y));
	}

	/**
	\brief returns MAX(x, y);
	*/
	EHD Type maxElement() const
	{
		return PxMax(x, y);
	}

	Type	x, y;
};

template<class Type>
static EHD PxVec2T<Type> operator*(Type f, const PxVec2T<Type>& v)
{
	return PxVec2T<Type>(f * v.x, f * v.y);
}

typedef PxVec2T<float>	PxVec2;
typedef PxVec2T<double>	PxVec2d;

} // namespace px

// ===== 원본: physx/include/foundation/PxVec3.h
namespace px
{

/**
\brief 3 Element vector class.

This is a 3-dimensional vector class with public data members.
*/
template<class Type>
class PxVec3T
{
  public:

	/**
	\brief default constructor leaves data uninitialized.
	*/
	EHD PxVec3T()
	{
	}

	/**
	\brief zero constructor.
	*/
	EHD PxVec3T(PxZERO) : x(Type(0.0)), y(Type(0.0)), z(Type(0.0))
	{
	}

	/**
	\brief Assigns scalar parameter to all elements.

	Useful to initialize to zero or one.

	\param[in] a Value to assign to elements.
	*/
	explicit EHD PxVec3T(Type a) : x(a), y(a), z(a)
	{
	}

	/**
	\brief Initializes from 3 scalar parameters.

	\param[in] nx Value to initialize X component.
	\param[in] ny Value to initialize Y component.
	\param[in] nz Value to initialize Z component.
	*/
	EHD PxVec3T(Type nx, Type ny, Type nz) : x(nx), y(ny), z(nz)
	{
	}

	/**
	\brief Copy ctor.
	*/
	EHD PxVec3T(const PxVec3T& v) : x(v.x), y(v.y), z(v.z)
	{
	}

	// Operators

	/**
	\brief Assignment operator
	*/
	EHD PxVec3T& operator=(const PxVec3T& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		return *this;
	}

	/**
	\brief element access
	*/
	EHD Type& operator[](unsigned int index)
	{
		E_ASSERT(index <= 2);
		return reinterpret_cast<Type*>(this)[index];
	}

	/**
	\brief element access
	*/
	EHD const Type& operator[](unsigned int index) const
	{
		E_ASSERT(index <= 2);
		return reinterpret_cast<const Type*>(this)[index];
	}

	/**
	\brief returns true if the two vectors are exactly equal.
	*/
	EHD bool operator==(const PxVec3T& v) const
	{
		return x == v.x && y == v.y && z == v.z;
	}

	/**
	\brief returns true if the two vectors are not exactly equal.
	*/
	EHD bool operator!=(const PxVec3T& v) const
	{
		return x != v.x || y != v.y || z != v.z;
	}

	/**
	\brief tests for exact zero vector
	*/
	EHD bool isZero() const
	{
		return x == Type(0.0) && y == Type(0.0) && z == Type(0.0);
	}

	/**
	\brief returns true if all 3 elems of the vector are finite (not NAN or INF, etc.)
	*/
	EHDI bool isFinite() const
	{
		return PxIsFinite(x) && PxIsFinite(y) && PxIsFinite(z);
	}

	/**
	\brief is normalized - used by API parameter validation
	*/
	EHD bool isNormalized() const
	{
		const float unitTolerance = Type(1e-4);	// PT: do we need a different epsilon for float & double?
		return isFinite() && PxAbs(magnitude() - Type(1.0)) < unitTolerance;
	}

	/**
	\brief returns the squared magnitude

	Avoids calling PxSqrt()!
	*/
	EHD Type magnitudeSquared() const
	{
		return x * x + y * y + z * z;
	}

	/**
	\brief returns the magnitude
	*/
	EHD Type magnitude() const
	{
		return PxSqrt(magnitudeSquared());
	}

	/**
	\brief negation
	*/
	EHD PxVec3T operator-() const
	{
		return PxVec3T(-x, -y, -z);
	}

	/**
	\brief vector addition
	*/
	EHD PxVec3T operator+(const PxVec3T& v) const
	{
		return PxVec3T(x + v.x, y + v.y, z + v.z);
	}

	/**
	\brief vector difference
	*/
	EHD PxVec3T operator-(const PxVec3T& v) const
	{
		return PxVec3T(x - v.x, y - v.y, z - v.z);
	}

	/**
	\brief scalar post-multiplication
	*/
	EHD PxVec3T operator*(Type f) const
	{
		return PxVec3T(x * f, y * f, z * f);
	}

	/**
	\brief scalar division
	*/
	EHD PxVec3T operator/(Type f) const
	{
		f = Type(1.0) / f;
		return PxVec3T(x * f, y * f, z * f);
	}

	/**
	\brief vector addition
	*/
	EHD PxVec3T& operator+=(const PxVec3T& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	/**
	\brief vector difference
	*/
	EHD PxVec3T& operator-=(const PxVec3T& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	/**
	\brief scalar multiplication
	*/
	EHD PxVec3T& operator*=(Type f)
	{
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}
	/**
	\brief scalar division
	*/
	EHD PxVec3T& operator/=(Type f)
	{
		f = Type(1.0) / f;
		x *= f;
		y *= f;
		z *= f;
		return *this;
	}

	/**
	\brief returns the scalar product of this and other.
	*/
	EHD Type dot(const PxVec3T& v) const
	{
		return x * v.x + y * v.y + z * v.z;
	}

	/**
	\brief cross product
	*/
	EHD PxVec3T cross(const PxVec3T& v) const
	{
		return PxVec3T(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}

	/** returns a unit vector */
	EHD PxVec3T getNormalized() const
	{
		const Type m = magnitudeSquared();
		return m > Type(0.0) ? *this * PxRecipSqrt(m) : PxVec3T(Type(0));
	}

	/**
	\brief normalizes the vector in place
	*/
	EHD Type normalize()
	{
		const Type m = magnitude();
		if(m > Type(0.0))
			*this /= m;
		return m;
	}

	/**
	\brief normalizes the vector in place. Does nothing if vector magnitude is under EPX_NORMALIZATION_EPSILON.
	Returns vector magnitude if >= EPX_NORMALIZATION_EPSILON and 0.0f otherwise.
	*/
	EHD Type normalizeSafe()
	{
		const Type mag = magnitude();
		if(mag < EPX_NORMALIZATION_EPSILON)	// PT: do we need a different epsilon for float & double?
			return Type(0.0);
		*this *= Type(1.0) / mag;
		return mag;
	}

	/**
	\brief normalizes the vector in place. Asserts if vector magnitude is under EPX_NORMALIZATION_EPSILON.
	returns vector magnitude.
	*/
	EHD Type normalizeFast()
	{
		const Type mag = magnitude();
		E_ASSERT(mag >= EPX_NORMALIZATION_EPSILON);	// PT: do we need a different epsilon for float & double?
		*this *= Type(1.0) / mag;
		return mag;
	}

	/**
	\brief a[i] * b[i], for all i.
	*/
	EHD PxVec3T multiply(const PxVec3T& a) const
	{
		return PxVec3T(x * a.x, y * a.y, z * a.z);
	}

	/**
	\brief element-wise minimum
	*/
	EHD PxVec3T minimum(const PxVec3T& v) const
	{
		return PxVec3T(PxMin(x, v.x), PxMin(y, v.y), PxMin(z, v.z));
	}

	/**
	\brief returns MIN(x, y, z);
	*/
	EHD Type minElement() const
	{
		return PxMin(x, PxMin(y, z));
	}

	/**
	\brief element-wise maximum
	*/
	EHD PxVec3T maximum(const PxVec3T& v) const
	{
		return PxVec3T(PxMax(x, v.x), PxMax(y, v.y), PxMax(z, v.z));
	}

	/**
	\brief returns MAX(x, y, z);
	*/
	EHD Type maxElement() const
	{
		return PxMax(x, PxMax(y, z));
	}

	/**
	\brief returns absolute values of components;
	*/
	EHD PxVec3T abs() const
	{
		return PxVec3T(PxAbs(x), PxAbs(y), PxAbs(z));
	}

	Type	x, y, z;
};

template<class Type>
static EHD PxVec3T<Type> operator*(Type f, const PxVec3T<Type>& v)
{
	return PxVec3T<Type>(f * v.x, f * v.y, f * v.z);
}

typedef PxVec3T<float>	PxVec3;
typedef PxVec3T<double>	PxVec3d;

//! A padded version of PxVec3, to safely load its data using SIMD
class PxVec3Padded : public PxVec3
{
	public:
	EHD	PxVec3Padded()								{}
	EHD	~PxVec3Padded()								{}
	EHD	PxVec3Padded(const PxVec3& p) : PxVec3(p)	{}
	EHD	PxVec3Padded(float f) : PxVec3(f)			{}

	/**
	\brief Assignment operator.
	To fix this:
	error: definition of implicit copy assignment operator for 'PxVec3Padded' is deprecated because it has a user-declared destructor [-Werror,-Wdeprecated]
	*/
	EHD PxVec3Padded& operator=(const PxVec3Padded& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		return *this;
	}

	PxU32	padding;
};
E_STATIC_ASSERT(sizeof(PxVec3Padded) == 16);

typedef PxVec3Padded	PxVec3p;

} // namespace px

// ===== 원본: physx/include/foundation/PxVec4.h
/**
\brief 4 Element vector class.

This is a 4-dimensional vector class with public data members.
*/
namespace px
{

template<class Type>
class PxVec4T
{
  public:
	/**
	\brief default constructor leaves data uninitialized.
	*/
	EHDI PxVec4T()
	{
	}

	/**
	\brief zero constructor.
	*/
	EHD PxVec4T(PxZERO) : x(Type(0.0)), y(Type(0.0)), z(Type(0.0)), w(Type(0.0))
	{
	}

	/**
	\brief Assigns scalar parameter to all elements.

	Useful to initialize to zero or one.

	\param[in] a Value to assign to elements.
	*/
	explicit EHDI PxVec4T(Type a) : x(a), y(a), z(a), w(a)
	{
	}

	/**
	\brief Initializes from 3 scalar parameters.

	\param[in] nx Value to initialize X component.
	\param[in] ny Value to initialize Y component.
	\param[in] nz Value to initialize Z component.
	\param[in] nw Value to initialize W component.
	*/
	EHDI PxVec4T(Type nx, Type ny, Type nz, Type nw) : x(nx), y(ny), z(nz), w(nw)
	{
	}

	/**
	\brief Initializes from 3 scalar parameters.

	\param[in] v Value to initialize the X, Y, and Z components.
	\param[in] nw Value to initialize W component.
	*/
	EHDI PxVec4T(const PxVec3T<Type>& v, Type nw) : x(v.x), y(v.y), z(v.z), w(nw)
	{
	}

	/**
	\brief Initializes from an array of scalar parameters.

	\param[in] v Value to initialize with.
	*/
	explicit EHDI PxVec4T(const Type v[]) : x(v[0]), y(v[1]), z(v[2]), w(v[3])
	{
	}

	/**
	\brief Copy ctor.
	*/
	EHDI PxVec4T(const PxVec4T& v) : x(v.x), y(v.y), z(v.z), w(v.w)
	{
	}

	// Operators

	/**
	\brief Assignment operator
	*/
	EHDI PxVec4T& operator=(const PxVec4T& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		w = p.w;
		return *this;
	}

	/**
	\brief element access
	*/
	EHDI Type& operator[](unsigned int index)
	{
		E_ASSERT(index <= 3);
		return reinterpret_cast<Type*>(this)[index];
	}

	/**
	\brief element access
	*/
	EHDI const Type& operator[](unsigned int index) const
	{
		E_ASSERT(index <= 3);
		return reinterpret_cast<const Type*>(this)[index];
	}

	/**
	\brief returns true if the two vectors are exactly equal.
	*/
	EHDI bool operator==(const PxVec4T& v) const
	{
		return x == v.x && y == v.y && z == v.z && w == v.w;
	}

	/**
	\brief returns true if the two vectors are not exactly equal.
	*/
	EHDI bool operator!=(const PxVec4T& v) const
	{
		return x != v.x || y != v.y || z != v.z || w != v.w;
	}

	/**
	\brief tests for exact zero vector
	*/
	EHDI bool isZero() const
	{
		return x == Type(0) && y == Type(0) && z == Type(0) && w == Type(0);
	}

	/**
	\brief returns true if all 3 elems of the vector are finite (not NAN or INF, etc.)
	*/
	EHDI bool isFinite() const
	{
		return PxIsFinite(x) && PxIsFinite(y) && PxIsFinite(z) && PxIsFinite(w);
	}

	/**
	\brief is normalized - used by API parameter validation
	*/
	EHDI bool isNormalized() const
	{
		const Type unitTolerance = Type(1e-4);
		return isFinite() && PxAbs(magnitude() - Type(1.0)) < unitTolerance;
	}

	/**
	\brief returns the squared magnitude

	Avoids calling PxSqrt()!
	*/
	EHDI Type magnitudeSquared() const
	{
		return x * x + y * y + z * z + w * w;
	}

	/**
	\brief returns the magnitude
	*/
	EHDI Type magnitude() const
	{
		return PxSqrt(magnitudeSquared());
	}

	/**
	\brief negation
	*/
	EHDI PxVec4T operator-() const
	{
		return PxVec4T(-x, -y, -z, -w);
	}

	/**
	\brief vector addition
	*/
	EHDI PxVec4T operator+(const PxVec4T& v) const
	{
		return PxVec4T(x + v.x, y + v.y, z + v.z, w + v.w);
	}

	/**
	\brief vector difference
	*/
	EHDI PxVec4T operator-(const PxVec4T& v) const
	{
		return PxVec4T(x - v.x, y - v.y, z - v.z, w - v.w);
	}

	/**
	\brief scalar post-multiplication
	*/
	EHDI PxVec4T operator*(Type f) const
	{
		return PxVec4T(x * f, y * f, z * f, w * f);
	}

	/**
	\brief scalar division
	*/
	EHDI PxVec4T operator/(Type f) const
	{
		f = Type(1.0) / f;
		return PxVec4T(x * f, y * f, z * f, w * f);
	}

	/**
	\brief vector addition
	*/
	EHDI PxVec4T& operator+=(const PxVec4T& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		w += v.w;
		return *this;
	}

	/**
	\brief vector difference
	*/
	EHDI PxVec4T& operator-=(const PxVec4T& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		w -= v.w;
		return *this;
	}

	/**
	\brief scalar multiplication
	*/
	EHDI PxVec4T& operator*=(Type f)
	{
		x *= f;
		y *= f;
		z *= f;
		w *= f;
		return *this;
	}
	/**
	\brief scalar division
	*/
	EHDI PxVec4T& operator/=(Type f)
	{
		f = Type(1.0) / f;
		x *= f;
		y *= f;
		z *= f;
		w *= f;
		return *this;
	}

	/**
	\brief returns the scalar product of this and other.
	*/
	EHDI Type dot(const PxVec4T& v) const
	{
		return x * v.x + y * v.y + z * v.z + w * v.w;
	}

	/** returns a unit vector */
	EHDI PxVec4T getNormalized() const
	{
		const Type m = magnitudeSquared();
		return m > Type(0.0) ? *this * PxRecipSqrt(m) : PxVec4T(Type(0));
	}

	/**
	\brief normalizes the vector in place
	*/
	EHDI Type normalize()
	{
		const Type m = magnitude();
		if(m > Type(0.0))
			*this /= m;
		return m;
	}

	/**
	\brief a[i] * b[i], for all i.
	*/
	EHDI PxVec4T multiply(const PxVec4T& a) const
	{
		return PxVec4T(x * a.x, y * a.y, z * a.z, w * a.w);
	}

	/**
	\brief element-wise minimum
	*/
	EHDI PxVec4T minimum(const PxVec4T& v) const
	{
		return PxVec4T(PxMin(x, v.x), PxMin(y, v.y), PxMin(z, v.z), PxMin(w, v.w));
	}

	/**
	\brief element-wise maximum
	*/
	EHDI PxVec4T maximum(const PxVec4T& v) const
	{
		return PxVec4T(PxMax(x, v.x), PxMax(y, v.y), PxMax(z, v.z), PxMax(w, v.w));
	}

	EHDI PxVec3T<Type> getXYZ() const
	{
		return PxVec3T<Type>(x, y, z);
	}

	Type	x, y, z, w;
};

template<class Type>
static EHDI PxVec4T<Type> operator*(Type f, const PxVec4T<Type>& v)
{
	return PxVec4T<Type>(f * v.x, f * v.y, f * v.z, f * v.w);
}

typedef PxVec4T<float>	PxVec4;
typedef PxVec4T<double>	PxVec4d;

} // namespace px

// ===== 원본: physx/include/foundation/PxQuat.h
namespace px
{

template<class Type> class PxMat33T;

/**
\brief This is a quaternion class. For more information on quaternion mathematics
consult a mathematics source on complex numbers.
*/

template<class Type>
class PxQuatT
{
  public:

	/**
	\brief Default constructor, does not do any initialization.
	*/
	EHD PxQuatT()
	{
	}

	//! identity constructor
	EHD PxQuatT(PxIDENTITY) : x(Type(0.0)), y(Type(0.0)), z(Type(0.0)), w(Type(1.0))
	{
	}

	/**
	\brief Constructor from a scalar: sets the real part w to the scalar value, and the imaginary parts (x,y,z) to zero
	*/
	explicit EHD PxQuatT(Type r) : x(Type(0.0)), y(Type(0.0)), z(Type(0.0)), w(r)
	{
	}

	/**
	\brief Constructor. Take note of the order of the elements!
	*/
	EHD PxQuatT(Type nx, Type ny, Type nz, Type nw) : x(nx), y(ny), z(nz), w(nw)
	{
	}

	/**
	\brief Creates from angle-axis representation.

	Axis must be normalized!

	Angle is in radians!

	<b>Unit:</b> Radians
	*/
	EHDI PxQuatT(Type angleRadians, const PxVec3T<Type>& unitAxis)
	{
		E_ASSERT(PxAbs(Type(1.0) - unitAxis.magnitude()) < Type(1e-3));
		const Type a = angleRadians * Type(0.5);

		Type s;
		PxSinCos(a, s, w);
		x = unitAxis.x * s;
		y = unitAxis.y * s;
		z = unitAxis.z * s;
	}

	/**
	\brief Copy ctor.
	*/
	EHD PxQuatT(const PxQuatT& v) : x(v.x), y(v.y), z(v.z), w(v.w)
	{
	}

	/**
	\brief Creates from orientation matrix.

	\param[in] m Rotation matrix to extract quaternion from.
	*/
	EHDI explicit PxQuatT(const PxMat33T<Type>& m); /* defined in PxMat33.h */

	/**
	\brief returns true if quat is identity
	*/
	EHD bool isIdentity() const
	{
		return x==Type(0.0) && y==Type(0.0) && z==Type(0.0) && w==Type(1.0);
	}

	/**
	\brief returns true if all elements are finite (not NAN or INF, etc.)
	*/
	bool isFinite() const
	{
		return PxIsFinite(x) && PxIsFinite(y) && PxIsFinite(z) && PxIsFinite(w);
	}

	/**
	\brief returns true if finite and magnitude is close to unit
	*/
	bool isUnit() const
	{
		const Type unitTolerance = Type(1e-3);
		return isFinite() && PxAbs(magnitude() - Type(1.0)) < unitTolerance;
	}

	/**
	\brief returns true if finite and magnitude is reasonably close to unit to allow for some accumulation of error vs
	isValid
	*/
	bool isSane() const
	{
		const Type unitTolerance = Type(1e-2);
		return isFinite() && PxAbs(magnitude() - Type(1.0)) < unitTolerance;
	}

	/**
	\brief returns true if the two quaternions are exactly equal
	*/
	EHD bool operator==(const PxQuatT& q) const
	{
		return x == q.x && y == q.y && z == q.z && w == q.w;
	}

	/**
	\brief converts this quaternion to angle-axis representation
	*/
	EHDI void toRadiansAndUnitAxis(Type& angle, PxVec3T<Type>& axis) const
	{
		const Type quatEpsilon = Type(1.0e-8);
		const Type s2 = x * x + y * y + z * z;
		if(s2 < quatEpsilon * quatEpsilon) // can't extract a sensible axis
		{
			angle = Type(0.0);
			axis = PxVec3T<Type>(Type(1.0), Type(0.0), Type(0.0));
		}
		else
		{
			const Type s = PxRecipSqrt(s2);
			axis = PxVec3T<Type>(x, y, z) * s;
			angle = PxAbs(w) < quatEpsilon ? Type(PxPi) : PxAtan2(s2 * s, w) * Type(2.0);
		}
	}

	/**
	\brief Gets the angle between this quat and the identity quaternion.

	<b>Unit:</b> Radians
	*/
	EHD Type getAngle() const
	{
		return PxAcos(w) * Type(2.0);
	}

	/**
	\brief Gets the angle between this quat and the argument

	<b>Unit:</b> Radians
	*/
	EHD Type getAngle(const PxQuatT& q) const
	{
		return PxAcos(dot(q)) * Type(2.0);
	}

	/**
	\brief This is the squared 4D vector length, should be 1 for unit quaternions.
	*/
	EHD Type magnitudeSquared() const
	{
		return x * x + y * y + z * z + w * w;
	}

	/**
	\brief returns the scalar product of this and other.
	*/
	EHD Type dot(const PxQuatT& v) const
	{
		return x * v.x + y * v.y + z * v.z + w * v.w;
	}

	EHD PxQuatT getNormalized() const
	{
		const Type s = Type(1.0) / magnitude();
		return PxQuatT(x * s, y * s, z * s, w * s);
	}

	EHD Type magnitude() const
	{
		return PxSqrt(magnitudeSquared());
	}

	// modifiers:
	/**
	\brief maps to the closest unit quaternion.
	*/
	EHD Type normalize() // convert this PxQuatT to a unit quaternion
	{
		const Type mag = magnitude();
		if(mag != Type(0.0))
		{
			const Type imag = Type(1.0) / mag;

			x *= imag;
			y *= imag;
			z *= imag;
			w *= imag;
		}
		return mag;
	}

	/*
	\brief returns the conjugate.

	\note for unit quaternions, this is the inverse.
	*/
	EHD PxQuatT getConjugate() const
	{
		return PxQuatT(-x, -y, -z, w);
	}

	/*
	\brief returns imaginary part.
	*/
	EHD PxVec3T<Type> getImaginaryPart() const
	{
		return PxVec3T<Type>(x, y, z);
	}

	/** brief computes rotation of x-axis */
	EHD PxVec3T<Type> getBasisVector0() const
	{
		const Type x2 = x * Type(2.0);
		const Type w2 = w * Type(2.0);
		return PxVec3T<Type>((w * w2) - Type(1.0) + x * x2, (z * w2) + y * x2, (-y * w2) + z * x2);
	}

	/** brief computes rotation of y-axis */
	EHD PxVec3T<Type> getBasisVector1() const
	{
		const Type y2 = y * Type(2.0);
		const Type w2 = w * Type(2.0);
		return PxVec3T<Type>((-z * w2) + x * y2, (w * w2) - Type(1.0) + y * y2, (x * w2) + z * y2);
	}

	/** brief computes rotation of z-axis */
	EHD PxVec3T<Type> getBasisVector2() const
	{
		const Type z2 = z * Type(2.0);
		const Type w2 = w * Type(2.0);
		return PxVec3T<Type>((y * w2) + x * z2, (-x * w2) + y * z2, (w * w2) - Type(1.0) + z * z2);
	}

	/**
	rotates passed vec by this (assumed unitary)
	*/
	EHD const PxVec3T<Type> rotate(const PxVec3T<Type>& v) const
	{
		const Type vx = Type(2.0) * v.x;
		const Type vy = Type(2.0) * v.y;
		const Type vz = Type(2.0) * v.z;
		const Type w2 = w * w - Type(0.5);
		const Type dot2 = (x * vx + y * vy + z * vz);
		return PxVec3T<Type>((vx * w2 + (y * vz - z * vy) * w + x * dot2), (vy * w2 + (z * vx - x * vz) * w + y * dot2),
						     (vz * w2 + (x * vy - y * vx) * w + z * dot2));
	}

	/** \brief computes inverse rotation of x-axis */
	EHD PxVec3T<Type> getInvBasisVector0() const
	{
		const Type x2 = x * Type(2.0);
		const Type w2 = w * Type(2.0);
		return PxVec3T<Type>((w * w2) - Type(1.0) + x * x2, (-z * w2) + y * x2, (y * w2) + z * x2);
	}

	/** \brief computes the inverse rotation of the y-axis */
	EHD PxVec3T<Type> getInvBasisVector1() const
	{
		const Type y2 = y * Type(2.0);
		const Type w2 = w * Type(2.0);
		return PxVec3T<Type>((z * w2) + x * y2, (w * w2) - Type(1.0) + y * y2, (-x * w2) + z * y2);
	}

	/** \brief computes the inverse rotation of the z-axis */
	EHD PxVec3T<Type> getInvBasisVector2() const
	{
		const Type z2 = z * Type(2.0);
		const Type w2 = w * Type(2.0);
		return PxVec3T<Type>((-y * w2) + x * z2, (x * w2) + y * z2, (w * w2) - Type(1.0) + z * z2);
	}

	/**
	inverse rotates passed vec by this (assumed unitary)
	*/
	EHD const PxVec3T<Type> rotateInv(const PxVec3T<Type>& v) const
	{
		const Type vx = Type(2.0) * v.x;
		const Type vy = Type(2.0) * v.y;
		const Type vz = Type(2.0) * v.z;
		const Type w2 = w * w - Type(0.5);
		const Type dot2 = (x * vx + y * vy + z * vz);
		return PxVec3T<Type>((vx * w2 - (y * vz - z * vy) * w + x * dot2), (vy * w2 - (z * vx - x * vz) * w + y * dot2),
						    (vz * w2 - (x * vy - y * vx) * w + z * dot2));
	}

	/**
	\brief Assignment operator
	*/
	EHD PxQuatT& operator=(const PxQuatT& p)
	{
		x = p.x;
		y = p.y;
		z = p.z;
		w = p.w;
		return *this;
	}

	EHD PxQuatT& operator*=(const PxQuatT& q)
	{
		const Type tx = w * q.x + q.w * x + y * q.z - q.y * z;
		const Type ty = w * q.y + q.w * y + z * q.x - q.z * x;
		const Type tz = w * q.z + q.w * z + x * q.y - q.x * y;

		w = w * q.w - q.x * x - y * q.y - q.z * z;
		x = tx;
		y = ty;
		z = tz;
		return *this;
	}

	EHD PxQuatT& operator+=(const PxQuatT& q)
	{
		x += q.x;
		y += q.y;
		z += q.z;
		w += q.w;
		return *this;
	}

	EHD PxQuatT& operator-=(const PxQuatT& q)
	{
		x -= q.x;
		y -= q.y;
		z -= q.z;
		w -= q.w;
		return *this;
	}

	EHD PxQuatT& operator*=(const Type s)
	{
		x *= s;
		y *= s;
		z *= s;
		w *= s;
		return *this;
	}

	/** quaternion multiplication */
	EHD PxQuatT operator*(const PxQuatT& q) const
	{
		return PxQuatT(w * q.x + q.w * x + y * q.z - q.y * z, w * q.y + q.w * y + z * q.x - q.z * x,
		              w * q.z + q.w * z + x * q.y - q.x * y, w * q.w - x * q.x - y * q.y - z * q.z);
	}

	/** quaternion addition */
	EHD PxQuatT operator+(const PxQuatT& q) const
	{
		return PxQuatT(x + q.x, y + q.y, z + q.z, w + q.w);
	}

	/** quaternion subtraction */
	EHD PxQuatT operator-() const
	{
		return PxQuatT(-x, -y, -z, -w);
	}

	EHD PxQuatT operator-(const PxQuatT& q) const
	{
		return PxQuatT(x - q.x, y - q.y, z - q.z, w - q.w);
	}

	EHD PxQuatT operator*(Type r) const
	{
		return PxQuatT(x * r, y * r, z * r, w * r);
	}

	/** the quaternion elements */
	Type	x, y, z, w;
};

typedef PxQuatT<float>	PxQuat;
typedef PxQuatT<double>	PxQuatd;

} // namespace px

// ===== 원본: physx/include/foundation/PxMat33.h
namespace px
{
/*!
\brief 3x3 matrix class

Some clarifications, as there have been much confusion about matrix formats etc in the past.

Short:
- Matrix have base vectors in columns (vectors are column matrices, 3x1 matrices).
- Matrix is physically stored in column major format
- Matrices are concaternated from left

Long:
Given three base vectors a, b and c the matrix is stored as

|a.x b.x c.x|
|a.y b.y c.y|
|a.z b.z c.z|

Vectors are treated as columns, so the vector v is

|x|
|y|
|z|

And matrices are applied _before_ the vector (pre-multiplication)
v' = M*v

|x'|   |a.x b.x c.x|   |x|   |a.x*x + b.x*y + c.x*z|
|y'| = |a.y b.y c.y| * |y| = |a.y*x + b.y*y + c.y*z|
|z'|   |a.z b.z c.z|   |z|   |a.z*x + b.z*y + c.z*z|


Physical storage and indexing:
To be compatible with popular 3d rendering APIs (read D3d and OpenGL)
the physical indexing is

|0 3 6|
|1 4 7|
|2 5 8|

index = column*3 + row

which in C++ translates to M[column][row]

The mathematical indexing is M_row,column and this is what is used for _-notation
so _12 is 1st row, second column and operator(row, column)!
*/

template<class Type>
class PxMat33T
{
	public:
	//! Default constructor
	EHD PxMat33T()
	{
	}

	//! identity constructor
	EHDI PxMat33T(PxIDENTITY) :
		column0(Type(1.0), Type(0.0), Type(0.0)),
		column1(Type(0.0), Type(1.0), Type(0.0)),
		column2(Type(0.0), Type(0.0), Type(1.0))
	{
	}

	//! zero constructor
	EHDI PxMat33T(PxZERO) :
		column0(Type(0.0)),
		column1(Type(0.0)),
		column2(Type(0.0))
	{
	}

	//! Construct from three base vectors
	PxMat33T(const PxVec3T<Type>& col0, const PxVec3T<Type>& col1, const PxVec3T<Type>& col2) :
		column0(col0),
		column1(col1),
		column2(col2)
	{
	}

	//! constructor from a scalar, which generates a multiple of the identity matrix
	explicit EHDI PxMat33T(Type r) :
		column0(r, Type(0.0), Type(0.0)),
		column1(Type(0.0), r, Type(0.0)),
		column2(Type(0.0), Type(0.0), r)
	{
	}

	//! Construct from Type[9]
	explicit EHDI PxMat33T(Type values[]) :
		column0(values[0], values[1], values[2]),
		column1(values[3], values[4], values[5]),
		column2(values[6], values[7], values[8])
	{
	}

	//! Construct from a quaternion
	explicit EHD PxMat33T(const PxQuatT<Type>& q)
	{
		// PT: TODO: PX-566
		const Type x = q.x;
		const Type y = q.y;
		const Type z = q.z;
		const Type w = q.w;

		const Type x2 = x + x;
		const Type y2 = y + y;
		const Type z2 = z + z;

		const Type xx = x2 * x;
		const Type yy = y2 * y;
		const Type zz = z2 * z;

		const Type xy = x2 * y;
		const Type xz = x2 * z;
		const Type xw = x2 * w;

		const Type yz = y2 * z;
		const Type yw = y2 * w;
		const Type zw = z2 * w;

		column0 = PxVec3T<Type>(Type(1.0) - yy - zz, xy + zw, xz - yw);
		column1 = PxVec3T<Type>(xy - zw, Type(1.0) - xx - zz, yz + xw);
		column2 = PxVec3T<Type>(xz + yw, yz - xw, Type(1.0) - xx - yy);
	}

	//! Copy constructor
	EHDI PxMat33T(const PxMat33T& other) :
		column0(other.column0),
		column1(other.column1),
		column2(other.column2)
	{
	}

	//! Assignment operator
	EHD PxMat33T& operator=(const PxMat33T& other)
	{
		column0 = other.column0;
		column1 = other.column1;
		column2 = other.column2;
		return *this;
	}

	//! Construct from diagonal, off-diagonals are zero.
	EHDI static const PxMat33T createDiagonal(const PxVec3T<Type>& d)
	{
		return PxMat33T(PxVec3T<Type>(d.x, Type(0.0), Type(0.0)),
						PxVec3T<Type>(Type(0.0), d.y, Type(0.0)),
						PxVec3T<Type>(Type(0.0), Type(0.0), d.z));
	}

	//! Computes the outer product of two vectors
	EHDI static const PxMat33T outer(const PxVec3T<Type>& a, const PxVec3T<Type>& b)
	{
		return PxMat33T(a * b.x, a * b.y, a * b.z);
	}

	/**
	\brief returns true if the two matrices are exactly equal
	*/
	EHDI bool operator==(const PxMat33T& m) const
	{
		return column0 == m.column0 && column1 == m.column1 && column2 == m.column2;
	}

	//! Get transposed matrix
	EHD const PxMat33T getTranspose() const
	{
		const PxVec3T<Type> v0(column0.x, column1.x, column2.x);
		const PxVec3T<Type> v1(column0.y, column1.y, column2.y);
		const PxVec3T<Type> v2(column0.z, column1.z, column2.z);

		return PxMat33T(v0, v1, v2);
	}

	//! Get the real inverse
	EHDI const PxMat33T getInverse() const
	{
		const Type det = getDeterminant();
		PxMat33T inverse;

		if(det != Type(0.0))
		{
			const Type invDet = Type(1.0) / det;

			inverse.column0.x = invDet * (column1.y * column2.z - column2.y * column1.z);
			inverse.column0.y = invDet * -(column0.y * column2.z - column2.y * column0.z);
			inverse.column0.z = invDet * (column0.y * column1.z - column0.z * column1.y);

			inverse.column1.x = invDet * -(column1.x * column2.z - column1.z * column2.x);
			inverse.column1.y = invDet * (column0.x * column2.z - column0.z * column2.x);
			inverse.column1.z = invDet * -(column0.x * column1.z - column0.z * column1.x);

			inverse.column2.x = invDet * (column1.x * column2.y - column1.y * column2.x);
			inverse.column2.y = invDet * -(column0.x * column2.y - column0.y * column2.x);
			inverse.column2.z = invDet * (column0.x * column1.y - column1.x * column0.y);

			return inverse;
		}
		else
		{
			return PxMat33T(PxIdentity);
		}
	}

	//! Get determinant
	EHDI Type getDeterminant() const
	{
		return column0.dot(column1.cross(column2));
	}

	//! Unary minus
	EHDI const PxMat33T operator-() const
	{
		return PxMat33T(-column0, -column1, -column2);
	}

	//! Add
	EHDI const PxMat33T operator+(const PxMat33T& other) const
	{
		return PxMat33T(column0 + other.column0, column1 + other.column1, column2 + other.column2);
	}

	//! Subtract
	EHDI const PxMat33T operator-(const PxMat33T& other) const
	{
		return PxMat33T(column0 - other.column0, column1 - other.column1, column2 - other.column2);
	}

	//! Scalar multiplication
	EHDI const PxMat33T operator*(Type scalar) const
	{
		return PxMat33T(column0 * scalar, column1 * scalar, column2 * scalar);
	}

	template<class Type2>
	EHDI friend PxMat33T<Type2> operator*(Type2, const PxMat33T<Type2>&);

	//! Matrix vector multiplication (returns 'this->transform(vec)')
	EHDI const PxVec3T<Type> operator*(const PxVec3T<Type>& vec) const
	{
		return transform(vec);
	}

	// a <op>= b operators

	//! Matrix multiplication
	EHD const PxMat33T operator*(const PxMat33T& other) const
	{
		// Rows from this <dot> columns from other
		// column0 = transform(other.column0) etc
		return PxMat33T(transform(other.column0),
						transform(other.column1),
						transform(other.column2));
	}

	//! Equals-add
	EHDI PxMat33T& operator+=(const PxMat33T& other)
	{
		column0 += other.column0;
		column1 += other.column1;
		column2 += other.column2;
		return *this;
	}

	//! Equals-sub
	EHDI PxMat33T& operator-=(const PxMat33T& other)
	{
		column0 -= other.column0;
		column1 -= other.column1;
		column2 -= other.column2;
		return *this;
	}

	//! Equals scalar multiplication
	EHDI PxMat33T& operator*=(Type scalar)
	{
		column0 *= scalar;
		column1 *= scalar;
		column2 *= scalar;
		return *this;
	}

	//! Equals matrix multiplication
	EHDI PxMat33T& operator*=(const PxMat33T& other)
	{
		*this = *this * other;
		return *this;
	}

	//! Element access, mathematical way!
	EHD Type operator()(PxU32 row, PxU32 col) const
	{
		return (*this)[col][row];
	}

	//! Element access, mathematical way!
	EHD Type& operator()(PxU32 row, PxU32 col)
	{
		return (*this)[col][row];
	}

	// Transform etc

	//! Transform vector by matrix, equal to v' = M*v
	EHD const PxVec3T<Type> transform(const PxVec3T<Type>& other) const
	{
		return column0 * other.x + column1 * other.y + column2 * other.z;
	}

	//! Transform vector by matrix transpose, v' = M^t*v
	EHDI const PxVec3T<Type> transformTranspose(const PxVec3T<Type>& other) const
	{
		return PxVec3T<Type>(column0.dot(other), column1.dot(other), column2.dot(other));
	}

	EHD const Type* front() const
	{
		return &column0.x;
	}

	EHD PxVec3T<Type>& operator[](PxU32 num)
	{
		return (&column0)[num];
	}

	EHD const PxVec3T<Type>& operator[](PxU32 num) const
	{
		return (&column0)[num];
	}

	// Data, see above for format!

	PxVec3T<Type>	column0, column1, column2; // the three base vectors
};

template<class Type>
EHDI PxMat33T<Type> operator*(Type scalar, const PxMat33T<Type>& m)
{
	return PxMat33T<Type>(scalar * m.column0, scalar * m.column1, scalar * m.column2);
}

// implementation from PxQuat.h
template<class Type>
EHDI PxQuatT<Type>::PxQuatT(const PxMat33T<Type>& m)
{
	if(m.column2.z < Type(0))
	{
		if(m.column0.x > m.column1.y)
		{
			const Type t = Type(1.0) + m.column0.x - m.column1.y - m.column2.z;
			*this = PxQuatT<Type>(t, m.column0.y + m.column1.x, m.column2.x + m.column0.z, m.column1.z - m.column2.y) * (Type(0.5) / PxSqrt(t));
		}
		else
		{
			const Type t = Type(1.0) - m.column0.x + m.column1.y - m.column2.z;
			*this = PxQuatT<Type>(m.column0.y + m.column1.x, t, m.column1.z + m.column2.y, m.column2.x - m.column0.z) * (Type(0.5) / PxSqrt(t));
		}
	}
	else
	{
		if(m.column0.x < -m.column1.y)
		{
			const Type t = Type(1.0) - m.column0.x - m.column1.y + m.column2.z;
			*this = PxQuatT<Type>(m.column2.x + m.column0.z, m.column1.z + m.column2.y, t, m.column0.y - m.column1.x) * (Type(0.5) / PxSqrt(t));
		}
		else
		{
			const Type t = Type(1.0) + m.column0.x + m.column1.y + m.column2.z;
			*this = PxQuatT<Type>(m.column1.z - m.column2.y, m.column2.x - m.column0.z, m.column0.y - m.column1.x, t) * (Type(0.5) / PxSqrt(t));
		}
	}
}

typedef PxMat33T<float>		PxMat33;
typedef PxMat33T<double>	PxMat33d;

	/**
	\brief Sets a rotation matrix around the X axis.
	\param m		[out] output rotation matrix
	\param angle	[in] desired angle
	*/
	EHDI	void PxSetRotX(PxMat33& m, PxReal angle)
	{
		m = PxMat33(PxIdentity);

		PxReal sin, cos;
		PxSinCos(angle, sin, cos);

		m[1][1] = m[2][2] = cos;
		m[1][2] = sin;
		m[2][1] = -sin;
	}

	/**
	\brief Sets a rotation matrix around the Y axis.
	\param m		[out] output rotation matrix
	\param angle	[in] desired angle
	*/
	EHDI	void PxSetRotY(PxMat33& m, PxReal angle)
	{
		m = PxMat33(PxIdentity);

		PxReal sin, cos;
		PxSinCos(angle, sin, cos);

		m[0][0] = m[2][2] = cos;
		m[0][2] = -sin;
		m[2][0] = sin;
	}

	/**
	\brief Sets a rotation matrix around the Z axis.
	\param m		[out] output rotation matrix
	\param angle	[in] desired angle
	*/
	EHDI	void PxSetRotZ(PxMat33& m, PxReal angle)
	{
		m = PxMat33(PxIdentity);

		PxReal sin, cos;
		PxSinCos(angle, sin, cos);

		m[0][0] = m[1][1] = cos;
		m[0][1] = sin;
		m[1][0] = -sin;
	}

	/**
	\brief Returns a rotation quaternion around the X axis.
	\param angle	[in] desired angle
	\return Quaternion that rotates around the desired axis
	*/
	EHDI	PxQuat PxGetRotXQuat(float angle)
	{
		PxMat33 m;
		PxSetRotX(m, angle);
		return PxQuat(m);
	}

	/**
	\brief Returns a rotation quaternion around the Y axis.
	\param angle	[in] desired angle
	\return Quaternion that rotates around the desired axis
	*/
	EHDI	PxQuat PxGetRotYQuat(float angle)
	{
		PxMat33 m;
		PxSetRotY(m, angle);
		return PxQuat(m);
	}

	/**
	\brief Returns a rotation quaternion around the Z axis.
	\param angle	[in] desired angle
	\return Quaternion that rotates around the desired axis
	*/
	EHDI	PxQuat PxGetRotZQuat(float angle)
	{
		PxMat33 m;
		PxSetRotZ(m, angle);
		return PxQuat(m);
	}

} // namespace px

// ===== 원본: physx/include/foundation/PxTransform.h
namespace px
{


template<class Type> class PxMat44T;

/*!
\brief class representing a rigid euclidean transform as a quaternion and a vector
*/

template<class Type>
class PxTransformT
{
  public:
	PxQuatT<Type>	q;
	PxVec3T<Type>	p;

	EHD PxTransformT()
	{
	}

	EHD explicit PxTransformT(PxIDENTITY) : q(PxIdentity), p(PxZero)
	{
	}

	EHD explicit PxTransformT(const PxVec3T<Type>& position) : q(PxIdentity), p(position)
	{
	}

	EHD explicit PxTransformT(const PxQuatT<Type>& orientation) : q(orientation), p(Type(0))
	{
		E_ASSERT(orientation.isSane());
	}

	EHD PxTransformT(Type x, Type y, Type z, PxQuatT<Type> aQ = PxQuatT<Type>(PxIdentity)) : q(aQ), p(x, y, z)
	{
	}

	EHD PxTransformT(const PxVec3T<Type>& p0, const PxQuatT<Type>& q0) : q(q0), p(p0)
	{
		E_ASSERT(q0.isSane());
	}

	EHD explicit PxTransformT(const PxMat44T<Type>& m); // defined in PxMat44.h

	EHD PxTransformT(const PxTransformT& other)
	{
		p = other.p;
		q = other.q;
	}

	EHD void operator=(const PxTransformT& other)
	{
		p = other.p;
		q = other.q;
	}

	/**
	\brief returns true if the two transforms are exactly equal
	*/
	EHDI bool operator==(const PxTransformT& t) const
	{
		return p == t.p && q == t.q;
	}

	EHD PxTransformT operator*(const PxTransformT& x) const
	{
		E_ASSERT(x.isSane());
		return transform(x);
	}

	//! Equals matrix multiplication
	EHDI PxTransformT& operator*=(const PxTransformT& other)
	{
		*this = *this * other;
		return *this;
	}

	EHD PxTransformT getInverse() const
	{
		E_ASSERT(isFinite());
		return PxTransformT(q.rotateInv(-p), q.getConjugate());
	}

	/**
	\brief return a normalized transform (i.e. one in which the quaternion has unit magnitude)
	*/
	EHD PxTransformT getNormalized() const
	{
		return PxTransformT(p, q.getNormalized());
	}

	EHD PxVec3T<Type> transform(const PxVec3T<Type>& input) const
	{
		E_ASSERT(isFinite());
		return q.rotate(input) + p;
	}

	EHD PxVec3T<Type> transformInv(const PxVec3T<Type>& input) const
	{
		E_ASSERT(isFinite());
		return q.rotateInv(input - p);
	}

	EHD PxVec3T<Type> rotate(const PxVec3T<Type>& input) const
	{
		E_ASSERT(isFinite());
		return q.rotate(input);
	}

	EHD PxVec3T<Type> rotateInv(const PxVec3T<Type>& input) const
	{
		E_ASSERT(isFinite());
		return q.rotateInv(input);
	}

	//! Transform transform to parent (returns compound transform: first src, then *this)
	EHD PxTransformT transform(const PxTransformT& src) const
	{
		E_ASSERT(src.isSane());
		E_ASSERT(isSane());
		// src = [srct, srcr] -> [r*srct + t, r*srcr]
		return PxTransformT(q.rotate(src.p) + p, q * src.q);
	}

	//! Transform transform from parent (returns compound transform: first src, then this->inverse)
	EHD PxTransformT transformInv(const PxTransformT& src) const
	{
		E_ASSERT(src.isSane());
		E_ASSERT(isFinite());
		// src = [srct, srcr] -> [r^-1*(srct-t), r^-1*srcr]
		const PxQuatT<Type> qinv = q.getConjugate();
		return PxTransformT(qinv.rotate(src.p - p), qinv * src.q);
	}

	/**
	\brief returns true if finite and q is a unit quaternion
	*/
	bool isValid() const
	{
		return p.isFinite() && q.isFinite() && q.isUnit();
	}

	/**
	\brief returns true if finite and quat magnitude is reasonably close to unit to allow for some accumulation of error
	vs isValid
	*/
	bool isSane() const
	{
		return isFinite() && q.isSane();
	}

	/**
	\brief returns true if all elems are finite (not NAN or INF, etc.)
	*/
	EHD bool isFinite() const
	{
		return p.isFinite() && q.isFinite();
	}
};

typedef PxTransformT<float>		PxTransform;
typedef PxTransformT<double>	PxTransformd;

/*!
\brief	A generic padded & aligned transform class.

This can be used for safe faster loads & stores, and faster address computations
(the default PxTransformT often generating imuls for this otherwise). Padding bytes
can be reused to store useful data if needed.
*/
struct  PxTransformPadded : PxTransform
{
	EHD PxTransformPadded()
	{
	}

	EHD PxTransformPadded(const PxTransformPadded& other) : PxTransform(other)
	{
	}

	EHD explicit PxTransformPadded(const PxTransform& other) : PxTransform(other)
	{
	}

	EHD explicit PxTransformPadded(PxIDENTITY) : PxTransform(PxIdentity)
	{
	}

	EHD explicit PxTransformPadded(const PxVec3& position) : PxTransform(position)
	{
	}

	EHD explicit PxTransformPadded(const PxQuat& orientation) : PxTransform(orientation)
	{
	}

	EHD PxTransformPadded(const PxVec3& p0, const PxQuat& q0) : PxTransform(p0, q0)
	{
	}

	EHD void operator=(const PxTransformPadded& other)
	{
		p = other.p;
		q = other.q;
	}

	EHD void operator=(const PxTransform& other)
	{
		p = other.p;
		q = other.q;
	}

	PxU32		padding;
}
;
E_STATIC_ASSERT(sizeof(PxTransformPadded)==32);

typedef PxTransformPadded	PxTransform32;

} // namespace px

// ===== 원본: physx/include/foundation/PxBounds3.h
namespace px
{

// maximum extents defined such that floating point exceptions are avoided for standard use cases
#define EPX_MAX_BOUNDS_EXTENTS (EPX_MAX_REAL * 0.25f)

/**
\brief Class representing 3D range or axis aligned bounding box.

Stored as minimum and maximum extent corners. Alternate representation
would be center and dimensions.
May be empty or nonempty. For nonempty bounds, minimum <= maximum has to hold for all axes.
Empty bounds have to be represented as minimum = EPX_MAX_BOUNDS_EXTENTS and maximum = -EPX_MAX_BOUNDS_EXTENTS for all
axes.
All other representations are invalid and the behavior is undefined.
*/
class PxBounds3
{
  public:
	/**
	\brief Default constructor, not performing any initialization for performance reason.
	\remark Use empty() function below to construct empty bounds.
	*/
	EHD PxBounds3()
	{
	}

	/**
	\brief Construct from two bounding points
	*/
	EHD PxBounds3(const PxVec3& minimum, const PxVec3& maximum);

	EHD void operator=(const PxBounds3& other)
	{
		minimum = other.minimum;
		maximum = other.maximum;
	}

	EHD PxBounds3(const PxBounds3& other)
	{
		minimum = other.minimum;
		maximum = other.maximum;
	}

	/**
	\brief Return empty bounds.
	*/
	static EHD PxBounds3 empty();

	/**
	\brief returns the AABB containing v0 and v1.
	\param v0 first point included in the AABB.
	\param v1 second point included in the AABB.
	*/
	static EHD PxBounds3 boundsOfPoints(const PxVec3& v0, const PxVec3& v1);

	/**
	\brief returns the AABB from center and extents vectors.
	\param center Center vector
	\param extent Extents vector
	*/
	static EHD PxBounds3 centerExtents(const PxVec3& center, const PxVec3& extent);

	/**
	\brief Construct from center, extent, and (not necessarily orthogonal) basis
	*/
	static EHDI PxBounds3 basisExtent(const PxVec3& center, const PxMat33& basis, const PxVec3& extent);

	/**
	\brief Construct from pose and extent
	*/
	static EHDI PxBounds3 poseExtent(const PxTransform& pose, const PxVec3& extent);

	/**
	\brief gets the transformed bounds of the passed AABB (resulting in a bigger AABB).

	This version is safe to call for empty bounds.

	\param[in] matrix Transform to apply, can contain scaling as well
	\param[in] bounds The bounds to transform.
	*/
	static EHDI PxBounds3 transformSafe(const PxMat33& matrix, const PxBounds3& bounds);

	/**
	\brief gets the transformed bounds of the passed AABB (resulting in a bigger AABB).

	Calling this method for empty bounds leads to undefined behavior. Use #transformSafe() instead.

	\param[in] matrix Transform to apply, can contain scaling as well
	\param[in] bounds The bounds to transform.
	*/
	static EHDI PxBounds3 transformFast(const PxMat33& matrix, const PxBounds3& bounds);

	/**
	\brief gets the transformed bounds of the passed AABB (resulting in a bigger AABB).

	This version is safe to call for empty bounds.

	\param[in] transform Transform to apply, can contain scaling as well
	\param[in] bounds The bounds to transform.
	*/
	static EHDI PxBounds3 transformSafe(const PxTransform& transform, const PxBounds3& bounds);

	/**
	\brief gets the transformed bounds of the passed AABB (resulting in a bigger AABB).

	Calling this method for empty bounds leads to undefined behavior. Use #transformSafe() instead.

	\param[in] transform Transform to apply, can contain scaling as well
	\param[in] bounds The bounds to transform.
	*/
	static EHDI PxBounds3 transformFast(const PxTransform& transform, const PxBounds3& bounds);

	/**
	\brief Sets empty to true
	*/
	EHD void setEmpty();

	/**
	\brief Sets the bounds to maximum size [-EPX_MAX_BOUNDS_EXTENTS, EPX_MAX_BOUNDS_EXTENTS].
	*/
	EHD void setMaximal();

	/**
	\brief expands the volume to include v
	\param v Point to expand to.
	*/
	EHD void include(const PxVec3& v);

	/**
	\brief expands the volume to include b.
	\param b Bounds to perform union with.
	*/
	EHD void include(const PxBounds3& b);

	EHD bool isEmpty() const;

	/**
	\brief indicates whether the intersection of this and b is empty or not.
	\param b Bounds to test for intersection.
	*/
	EHD bool intersects(const PxBounds3& b) const;

	/**
	 \brief computes the 1D-intersection between two AABBs, on a given axis.
	 \param	a		the other AABB
	 \param	axis	the axis (0, 1, 2)
	 */
	EHD bool intersects1D(const PxBounds3& a, uint32_t axis) const;

	/**
	\brief indicates if these bounds contain v.
	\param v Point to test against bounds.
	*/
	EHD bool contains(const PxVec3& v) const;

	/**
	 \brief	checks a box is inside another box.
	 \param	box		the other AABB
	 */
	EHD bool isInside(const PxBounds3& box) const;

	/**
	\brief returns the center of this axis aligned box.
	*/
	EHD PxVec3 getCenter() const;

	/**
	\brief get component of the box's center along a given axis
	*/
	EHD float getCenter(uint32_t axis) const;

	/**
	\brief get component of the box's extents along a given axis
	*/
	EHD float getExtents(uint32_t axis) const;

	/**
	\brief returns the dimensions (width/height/depth) of this axis aligned box.
	*/
	EHD PxVec3 getDimensions() const;

	/**
	\brief returns the extents, which are half of the width/height/depth.
	*/
	EHD PxVec3 getExtents() const;

	/**
	\brief scales the AABB.

	This version is safe to call for empty bounds.

	\param scale Factor to scale AABB by.
	*/
	EHD void scaleSafe(float scale);

	/**
	\brief scales the AABB.

	Calling this method for empty bounds leads to undefined behavior. Use #scaleSafe() instead.

	\param scale Factor to scale AABB by.
	*/
	EHD void scaleFast(float scale);

	/**
	fattens the AABB in all 3 dimensions by the given distance.

	This version is safe to call for empty bounds.
	*/
	EHD void fattenSafe(float distance);

	/**
	fattens the AABB in all 3 dimensions by the given distance.

	Calling this method for empty bounds leads to undefined behavior. Use #fattenSafe() instead.
	*/
	EHD void fattenFast(float distance);

	/**
	checks that the AABB values are not NaN
	*/
	EHD bool isFinite() const;

	/**
	checks that the AABB values describe a valid configuration.
	*/
	EHD bool isValid() const;

	/**
	Finds the closest point in the box to the point p. If p is contained, this will be p, otherwise it 
	will be the closest point on the surface of the box.
	*/
	EHD PxVec3 closestPoint(const PxVec3& p) const;

	PxVec3 minimum, maximum;
};

EHD PxBounds3::PxBounds3(const PxVec3& minimum_, const PxVec3& maximum_)
: minimum(minimum_), maximum(maximum_)
{
}

EHD PxBounds3 PxBounds3::empty()
{
	return PxBounds3(PxVec3(EPX_MAX_BOUNDS_EXTENTS), PxVec3(-EPX_MAX_BOUNDS_EXTENTS));
}

EHD bool PxBounds3::isFinite() const
{
	return minimum.isFinite() && maximum.isFinite();
}

EHD PxBounds3 PxBounds3::boundsOfPoints(const PxVec3& v0, const PxVec3& v1)
{
	return PxBounds3(v0.minimum(v1), v0.maximum(v1));
}

EHD PxBounds3 PxBounds3::centerExtents(const PxVec3& center, const PxVec3& extent)
{
	return PxBounds3(center - extent, center + extent);
}

EHDI PxBounds3
PxBounds3::basisExtent(const PxVec3& center, const PxMat33& basis, const PxVec3& extent)
{
	// extended basis vectors
	const PxVec3 c0 = basis.column0 * extent.x;
	const PxVec3 c1 = basis.column1 * extent.y;
	const PxVec3 c2 = basis.column2 * extent.z;

	// find combination of base vectors that produces max. distance for each component = sum of abs()
	const PxVec3 w(	PxAbs(c0.x) + PxAbs(c1.x) + PxAbs(c2.x),
					PxAbs(c0.y) + PxAbs(c1.y) + PxAbs(c2.y),
					PxAbs(c0.z) + PxAbs(c1.z) + PxAbs(c2.z));

	return PxBounds3(center - w, center + w);
}

EHDI PxBounds3 PxBounds3::poseExtent(const PxTransform& pose, const PxVec3& extent)
{
	return basisExtent(pose.p, PxMat33(pose.q), extent);
}

EHD void PxBounds3::setEmpty()
{
	minimum = PxVec3(EPX_MAX_BOUNDS_EXTENTS);
	maximum = PxVec3(-EPX_MAX_BOUNDS_EXTENTS);
}

EHD void PxBounds3::setMaximal()
{
	minimum = PxVec3(-EPX_MAX_BOUNDS_EXTENTS);
	maximum = PxVec3(EPX_MAX_BOUNDS_EXTENTS);
}

EHD void PxBounds3::include(const PxVec3& v)
{
	E_ASSERT(isValid());
	minimum = minimum.minimum(v);
	maximum = maximum.maximum(v);
}

EHD void PxBounds3::include(const PxBounds3& b)
{
	E_ASSERT(isValid());
	minimum = minimum.minimum(b.minimum);
	maximum = maximum.maximum(b.maximum);
}

EHD bool PxBounds3::isEmpty() const
{
	E_ASSERT(isValid());
	return minimum.x > maximum.x;
}

EHD bool PxBounds3::intersects(const PxBounds3& b) const
{
	E_ASSERT(isValid() && b.isValid());
	return !(b.minimum.x > maximum.x || minimum.x > b.maximum.x || b.minimum.y > maximum.y || minimum.y > b.maximum.y ||
	         b.minimum.z > maximum.z || minimum.z > b.maximum.z);
}

EHD bool PxBounds3::intersects1D(const PxBounds3& a, uint32_t axis) const
{
	E_ASSERT(isValid() && a.isValid());
	return maximum[axis] >= a.minimum[axis] && a.maximum[axis] >= minimum[axis];
}

EHD bool PxBounds3::contains(const PxVec3& v) const
{
	E_ASSERT(isValid());

	return !(v.x < minimum.x || v.x > maximum.x || v.y < minimum.y || v.y > maximum.y || v.z < minimum.z ||
	         v.z > maximum.z);
}

EHD bool PxBounds3::isInside(const PxBounds3& box) const
{
	E_ASSERT(isValid() && box.isValid());
	if(box.minimum.x > minimum.x)
		return false;
	if(box.minimum.y > minimum.y)
		return false;
	if(box.minimum.z > minimum.z)
		return false;
	if(box.maximum.x < maximum.x)
		return false;
	if(box.maximum.y < maximum.y)
		return false;
	if(box.maximum.z < maximum.z)
		return false;
	return true;
}

EHD PxVec3 PxBounds3::getCenter() const
{
	E_ASSERT(isValid());
	return (minimum + maximum) * 0.5f;
}

EHD float PxBounds3::getCenter(uint32_t axis) const
{
	E_ASSERT(isValid());
	return (minimum[axis] + maximum[axis]) * 0.5f;
}

EHD float PxBounds3::getExtents(uint32_t axis) const
{
	E_ASSERT(isValid());
	return (maximum[axis] - minimum[axis]) * 0.5f;
}

EHD PxVec3 PxBounds3::getDimensions() const
{
	E_ASSERT(isValid());
	return maximum - minimum;
}

EHD PxVec3 PxBounds3::getExtents() const
{
	E_ASSERT(isValid());
	return getDimensions() * 0.5f;
}

EHD void PxBounds3::scaleSafe(float scale)
{
	E_ASSERT(isValid());
	if(!isEmpty())
		scaleFast(scale);
}

EHD void PxBounds3::scaleFast(float scale)
{
	E_ASSERT(isValid());
	*this = centerExtents(getCenter(), getExtents() * scale);
}

EHD void PxBounds3::fattenSafe(float distance)
{
	E_ASSERT(isValid());
	if(!isEmpty())
		fattenFast(distance);
}

EHD void PxBounds3::fattenFast(float distance)
{
	E_ASSERT(isValid());
	minimum.x -= distance;
	minimum.y -= distance;
	minimum.z -= distance;

	maximum.x += distance;
	maximum.y += distance;
	maximum.z += distance;
}

EHDI PxBounds3 PxBounds3::transformSafe(const PxMat33& matrix, const PxBounds3& bounds)
{
	E_ASSERT(bounds.isValid());
	return !bounds.isEmpty() ? transformFast(matrix, bounds) : bounds;
}

EHDI PxBounds3 PxBounds3::transformFast(const PxMat33& matrix, const PxBounds3& bounds)
{
	E_ASSERT(bounds.isValid());
	return PxBounds3::basisExtent(matrix * bounds.getCenter(), matrix, bounds.getExtents());
}

EHDI PxBounds3 PxBounds3::transformSafe(const PxTransform& transform, const PxBounds3& bounds)
{
	E_ASSERT(bounds.isValid());
	return !bounds.isEmpty() ? transformFast(transform, bounds) : bounds;
}

EHDI PxBounds3 PxBounds3::transformFast(const PxTransform& transform, const PxBounds3& bounds)
{
	E_ASSERT(bounds.isValid());
	return PxBounds3::basisExtent(transform.transform(bounds.getCenter()), PxMat33(transform.q), bounds.getExtents());
}

EHD bool PxBounds3::isValid() const
{
	return (isFinite() && (((minimum.x <= maximum.x) && (minimum.y <= maximum.y) && (minimum.z <= maximum.z)) ||
	                       ((minimum.x == EPX_MAX_BOUNDS_EXTENTS) && (minimum.y == EPX_MAX_BOUNDS_EXTENTS) &&
	                        (minimum.z == EPX_MAX_BOUNDS_EXTENTS) && (maximum.x == -EPX_MAX_BOUNDS_EXTENTS) &&
	                        (maximum.y == -EPX_MAX_BOUNDS_EXTENTS) && (maximum.z == -EPX_MAX_BOUNDS_EXTENTS))));
}

EHD PxVec3 PxBounds3::closestPoint(const PxVec3& p) const
{
	return minimum.maximum(maximum.minimum(p));
}

} // namespace px

// ===== 원본: physx/include/foundation/PxPlane.h
namespace px
{

/**
\brief Representation of a plane.

 Plane equation used: n.dot(v) + d = 0
*/
class PxPlane
{
  public:
	/**
	\brief Constructor
	*/
	EHD PxPlane()
	{
	}

	/**
	\brief Constructor from a normal and a distance
	*/
	EHD PxPlane(float nx, float ny, float nz, float distance) : n(nx, ny, nz), d(distance)
	{
	}

	/**
	\brief Constructor from a normal and a distance
	*/
	EHD PxPlane(const PxVec3& normal, float distance) : n(normal), d(distance)
	{
	}

	/**
	\brief Constructor from a point on the plane and a normal
	*/
	EHD PxPlane(const PxVec3& point, const PxVec3& normal)
	: n(normal), d(-point.dot(n)) // p satisfies normal.dot(p) + d = 0
	{
	}

	/**
	\brief Constructor from three points
	*/
	EHD PxPlane(const PxVec3& p0, const PxVec3& p1, const PxVec3& p2)
	{
		n = (p1 - p0).cross(p2 - p0).getNormalized();
		d = -p0.dot(n);
	}

	/**
	\brief returns true if the two planes are exactly equal
	*/
	EHDI bool operator==(const PxPlane& p) const
	{
		return n == p.n && d == p.d;
	}

	EHD float distance(const PxVec3& p) const
	{
		return p.dot(n) + d;
	}

	EHD bool contains(const PxVec3& p) const
	{
		return PxAbs(distance(p)) < (1.0e-7f);
	}

	/**
	\brief projects p into the plane
	*/
	EHD PxVec3 project(const PxVec3& p) const
	{
		return p - n * distance(p);
	}

	/**
	\brief find an arbitrary point in the plane
	*/
	EHD PxVec3 pointInPlane() const
	{
		return -n * d;
	}

	/**
	\brief equivalent plane with unit normal
	*/

	EHD void normalize()
	{
		float denom = 1.0f / n.magnitude();
		n *= denom;
		d *= denom;
	}

	/**
	\brief transform plane
	*/
	EHD PxPlane transform(const PxTransform& pose) const
	{
		const PxVec3 transformedNormal = pose.rotate(n);
		return PxPlane(transformedNormal, d - pose.p.dot(transformedNormal));
	}

	/**
	\brief inverse-transform plane
	*/
	EHD PxPlane inverseTransform(const PxTransform& pose) const
	{
		const PxVec3 transformedNormal = pose.rotateInv(n);
		return PxPlane(transformedNormal, d + pose.p.dot(n));
	}

	PxVec3 n; //!< The normal to the plane
	float d;  //!< The distance from the origin
};

} // namespace px

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using UnityEditor;


namespace DNAMath
{
    /// <summary>
    /// Custom Matrix Class developped for working on the GPU and with DNANeuralNetworks
    /// </summary>
    public class DNAMatrix
    {
        // 0--------> Width
        // |
        // |
        // |
        // Height

        /// <summary>
        /// Shader Script that runs Matrix Multiplication on the GPU
        /// </summary>
        public static ComputeShader matrixMultScript;

        /// <summary>
        /// Shader Script that runs Matrix Addition on the GPU
        /// </summary>
        public static ComputeShader matrixAdditionScript;

        /// <summary>
        /// Shader Script that runs Matrix Substraction on the GPU
        /// </summary>
        public static ComputeShader matrixSubstractionScript;

        /// <summary>
        /// Load the Shader Scripts associated for speeding up the mathematics
        /// </summary>
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterAssembliesLoaded)]
        public static void loadAssets()
        {
            matrixMultScript = AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/DNAMath/MatrixMultiplicationGPU.compute");
            matrixAdditionScript = AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/DNAMath/MatrixAdditionGPU.compute");
            matrixSubstractionScript = AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/DNAMath/MatrixSubstractionGPU.compute");
        }

        /// <summary>
        /// Describes the number of rows the matrix has
        /// </summary>
        private int _height;

        /// <summary>
        /// Describes the number of rows the matrix has
        /// </summary>
        public int Height
        {
            get
            {
                return _height;
            }
        }

        /// <summary>
        /// Describes the number of columns the matrix has
        /// </summary>
        private int _width;

        /// <summary>
        /// Describes the number of columns the matrix has
        /// </summary>
        public int Width
        {
            get
            {
                return _width;
            }
        }

        /// <summary>
        /// A list of all values contained in the matrix
        /// </summary>
        private double[] _values;

        /// <summary>
        /// A list of all values contained in the matrix
        /// </summary>
        public double[] Values
        {
            get
            {
                return _values;
            }
            set
            {
                _values = value;
            }
        }

        /// <summary>
        /// Constructor function initializing the Matrix
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        public DNAMatrix(int height, int width)
        {
            this._width = width;
            this._height = height;

            _values = new double[width * height];
        }

        /// <summary>
        /// Indexer allowing us to get access and sey  to a value using array notation
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <returns></returns>
        public double this[int height, int width]
        {
            get
            {
                return _values[GetFlatIndex(height, width)];
            }
            set
            {
                _values[GetFlatIndex(height, width)] = value;
            }
        }

        /// <summary>
        /// Indexer allowing us to access and set the value in array notation
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double this[int index]
        {
            get
            {
                return _values[index];
            }
            set
            {
                _values[index] = value;
            }
        }

        /// <summary>
        /// Indexer allowing us to get a vector from the matrix
        /// </summary>
        /// <param name="index"></param>
        /// <param name="idk"></param>
        /// <returns></returns>
        public double[] this[int index, bool row]
        {
            get
            {
                double[] vector;
                if (row)
                {
                    //Get a row
                    vector = new double[Width];

                    for (int i = 0; i < Width; i++)
                    {
                        vector[i] = this[index, i];
                    }

                }
                else
                {
                    //Get a column
                    vector = new double[Height];

                    for (int i = 0; i < Height; i++)
                    {
                        vector[i] = this[i, index];
                    }
                }

                return vector;
            }
        }


        /// <summary>
        /// Initializes a matrix that counts from 1 - the number of values based on the dimensions
        /// </summary>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <returns></returns>
        public static DNAMatrix Increment(int height, int width)
        {
            DNAMatrix matrix = new DNAMatrix(height, width);

            for (int i = 0; i < width * height; i++)
            {
                matrix[i] = i + 1;
            }

            return matrix;
        }

        /// <summary>
        /// Sets the value at the given height and width index
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <param name="val"></param>
        public void SetValue(int heightIndex, int widthIndex, double val)
        {
            this._values[GetFlatIndex(heightIndex, widthIndex)] = val;
        }

        /// <summary>
        /// Add to the values at the given height and width index
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <param name="val"></param>
        public void AddValue(int heightIndex, int widthIndex, double val)
        {
            this._values[GetFlatIndex(heightIndex, widthIndex)] += val;
        }

        /// <summary>
        /// Gets the value at the given height and width index
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <returns></returns>
        public double GetValue(int heightIndex, int widthIndex)
        {
            return _values[GetFlatIndex(heightIndex, widthIndex)];
        }

        /// <summary>
        /// Returns the flat index of a value 
        /// </summary>
        /// <param name="heightIndex"></param>
        /// <param name="widthIndex"></param>
        /// <returns></returns>
        public int GetFlatIndex(int heightIndex, int widthIndex)
        {
            return heightIndex * Width + widthIndex;
        }



        /// <summary>
        /// Returns calculated indeces for the matrix based on a length index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public (int height, int width) GetIndex(int index)
        {
            int height = index / Width;
            int width = index % Width;

            return (height, width);
        }

        /// <summary>
        /// Returns calculated indeces for the matrix based on a length index for static functions
        /// </summary>
        /// <param name="index"></param>
        /// <param name="mat"></param>
        /// <returns></returns>
        public static (int height, int width) GetIndex(int index, DNAMatrix mat)
        {
            int height = index / mat.Width;
            int width = index % mat.Width;

            return (height, width);
        }

        /// <summary>
        /// Returns the Dot Product
        /// </summary>
        /// <param name="vector1"></param>
        /// <param name="vector2"></param>
        /// <returns></returns>
        double DotProduct(double[] vector1, double[] vector2)
        {
            double value = 0;

            if (vector1.Length == vector2.Length)
            {
                for (int i = 0; i < vector1.Length; i++)
                {
                    value += vector1[i] * vector2[i];
                }
            }

            return value;
        }

        /// <summary>
        /// Returns the Dot Product for Static Functions
        /// </summary>
        /// <param name="vector1"></param>
        /// <param name="vector2"></param>
        /// <returns></returns>
        static double DotProductStatic(double[] vector1, double[] vector2)
        {
            double value = 0;

            if (vector1.Length == vector2.Length)
            {
                for (int i = 0; i < vector1.Length; i++)
                {
                    value += vector1[i] * vector2[i];
                }
            }

            return value;
        }

        /*
        /// <summary>
        /// Transposes the matrix
        /// </summary>
        public void Transpose()
        {
            DNAMatrix newMat = new DNAMatrix(this.Width, this.Height);

            for (int width = 0; width < this.Width; width++)
            {
                for (int height = 0; height < this.Height; height++)
                {
                    newMat[width, height] = this[height, width];
                }
            }

            this._width = newMat.Width;
            this._height = newMat.Height;
            this._values = newMat.Values;
        }
        */

        /// <summary>
        /// Transposes the matrix
        /// </summary>
        public DNAMatrix Transpose()
        {

            DNAMatrix newMat = new DNAMatrix(this.Width, this.Height);

            for (int width = 0; width < this.Width; width++)
            {
                for (int height = 0; height < this.Height; height++)
                {
                    newMat[width, height] = this[height, width];
                }
            }

            return newMat;
        }

        /// <summary>
        /// Operator for adding two matrices together
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static DNAMatrix operator +(DNAMatrix matrixA, DNAMatrix matrixB)
        {
            DNAMatrix newMat = new DNAMatrix(0, 0);

            if (matrixAdditionScript != null)
                newMat = matrixAdditionGPU(matrixA, matrixB);
            else
            {
                if (matrixA.Height == matrixB.Height && matrixA.Width == matrixB.Width)
                {
                    newMat = new DNAMatrix(matrixA.Height, matrixA.Width);

                    for (int i = 0; i < matrixA.Values.Length; i++)
                    {
                        newMat[i] = matrixA[i] + matrixB[i];
                    }
                }
                else
                {
                    Debug.Log("Error, Dimensions don't match");
                }
            }
           
            return newMat;
        }

        /// <summary>
        /// Operator handling subtractions between 2 matrices
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static DNAMatrix operator -(DNAMatrix matrixA, DNAMatrix matrixB)
        {
            DNAMatrix newMat = new DNAMatrix(0, 0);

            if (matrixA.Height == matrixB.Height && matrixA.Width == matrixB.Width)
            {
                newMat = new DNAMatrix(matrixA.Height, matrixA.Width);

                for (int i = 0; i < matrixA.Values.Length; i++)
                {
                    newMat[i] = matrixA[i] - matrixB[i];
                }
            }
            else
            {
                Debug.Log("Error, Dimensions don't match");
            }

            return newMat;
        }

        /// <summary>
        /// Multiplication operation, multiplying matrices together
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static DNAMatrix operator *(DNAMatrix matrixA, DNAMatrix matrixB)
        {
            DNAMatrix newMat = new DNAMatrix(0, 0);

            if (matrixMultScript != null)
                newMat = multMatrixGPU(matrixA, matrixB);
            else
            {
                //Check if matrixA Width is equal to matrixB Height
                if (matrixA.Width == matrixB.Height)
                {
                    newMat = new DNAMatrix(matrixA.Height, matrixB.Width);

                    for (int index = 0; index < newMat.Values.Length; index++)
                    {
                        (int height, int width) = GetIndex(index, newMat);

                        newMat[index] = DotProductStatic(matrixA[height, true], matrixB[width, false]);
                    }
                }
                else
                    Debug.Log("Error, Dimensions don't match");
            }

            return newMat;
        }

        /// <summary>
        /// Multiplication operation with a factor
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="factor"></param>
        /// <returns></returns>
        public static DNAMatrix operator *(DNAMatrix matrixA, double factor)
        {
            DNAMatrix newMat = new DNAMatrix(0, 0);

            newMat = new DNAMatrix(matrixA.Height, matrixA.Width);

            for (int i = 0; i < matrixA.Values.Length; i++)
            {
                newMat[i] = matrixA[i] * factor;
            }

            return newMat;
        }

        /// <summary>
        /// Handles a Matrix Multiplication by handing it off to the GPU, this makes it crazy fast
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static DNAMatrix multMatrixGPU(DNAMatrix matrixA, DNAMatrix matrixB)
        {
            DNAMatrix newMat = new DNAMatrix(0, 0);
            if (matrixA.Width == matrixB.Height)
            {
                newMat = new DNAMatrix(matrixA.Height, matrixB.Width);

                ComputeShader compShader = matrixMultScript;

                ComputeBuffer matrixAVals = new ComputeBuffer(matrixA.Values.Length, sizeof(double));
                matrixAVals.SetData(matrixA.Values);

                ComputeBuffer matrixBVals = new ComputeBuffer(matrixB.Values.Length, sizeof(double));
                matrixBVals.SetData(matrixB.Values);

                ComputeBuffer newMatVals = new ComputeBuffer(newMat.Values.Length, sizeof(double));
                newMatVals.SetData(newMat.Values);

                //Convert dimensions to array of int
                int[] dim1 = new int[2];
                dim1[0] = matrixA.Height;
                dim1[1] = matrixA.Width;
                ComputeBuffer matrixADim = new ComputeBuffer(2, sizeof(int));
                matrixADim.SetData(dim1);

                int[] dim2 = new int[2];
                dim2[0] = matrixB.Height;
                dim2[1] = matrixB.Width;
                ComputeBuffer matrixBDim = new ComputeBuffer(2, sizeof(int));
                matrixBDim.SetData(dim2);

                int[] newDim = new int[2];
                newDim[0] = newMat.Height;
                newDim[1] = newMat.Width;
                ComputeBuffer newMatDim = new ComputeBuffer(2, sizeof(int));
                newMatDim.SetData(newDim);

                //Set the buffer info
                compShader.SetBuffer(0, "matrixAVals", matrixAVals);
                compShader.SetBuffer(0, "matrixBVals", matrixBVals);
                compShader.SetBuffer(0, "newMatVals", newMatVals);
                compShader.SetBuffer(0, "matrixADim", matrixADim);
                compShader.SetBuffer(0, "matrixBDim", matrixBDim);
                compShader.SetBuffer(0, "newMatDim", newMatDim);

                System.DateTime startTime;
                System.DateTime endTime;

                startTime = System.DateTime.UtcNow;

                int groupCount = 0;
                if ((newMat.Values.Length / 1024) >= 1)
                {
                    groupCount = (newMat.Values.Length / 1024);
                }
                else
                {
                    groupCount = 1;
                }

                //Calculate
                compShader.Dispatch(0, groupCount, 1, 1);

                //Receaive Result
                newMatVals.GetData(newMat.Values);

                //Get rid of memory
                matrixAVals.Dispose();
                matrixBVals.Dispose();
                newMatVals.Dispose();

                matrixADim.Dispose();
                matrixBDim.Dispose();
                newMatDim.Dispose();

                endTime = System.DateTime.UtcNow;

                //Debug.Log("Total Time elapsed GPU (Matrix Multiplication) (" + matrixA.Height + "x" + matrixA.Width + "*" + matrixB.Height + "x" + matrixB.Width + "): " + (endTime - startTime).TotalMilliseconds + "(MilliSeconds)");
            } else
                Debug.Log("Error, Dimensions don't match");

            return newMat;
        }

        /// <summary>
        /// Handles Matrix Additions through the GPU
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public static DNAMatrix matrixAdditionGPU(DNAMatrix matrixA, DNAMatrix matrixB)
        {
            DNAMatrix newMat = new DNAMatrix(0, 0);

            if (_SameDimension(matrixA, matrixB))
            {
                ComputeShader computeShader = matrixAdditionScript;
                newMat = new DNAMatrix(matrixA.Height, matrixB.Width);

                ComputeBuffer matrixAValues = new ComputeBuffer(matrixA.Values.Length, sizeof(double));
                matrixAValues.SetData(matrixA.Values);

                ComputeBuffer matrixBValues = new ComputeBuffer(matrixB.Values.Length, sizeof(double));
                matrixBValues.SetData(matrixB.Values);

                ComputeBuffer newMatrixValues = new ComputeBuffer(newMat.Values.Length, sizeof(double));
                newMatrixValues.SetData(newMat.Values);

                computeShader.SetBuffer(0, "matrixAValues", matrixAValues);
                computeShader.SetBuffer(0, "matrixBValues", matrixBValues);
                computeShader.SetBuffer(0, "newMatrixValues", newMatrixValues);

                int groupCount = 0;
                if ((newMat.Values.Length / 1024) >= 1)
                {
                    groupCount = (newMat.Values.Length / 1024);
                }
                else
                {
                    groupCount = 1;
                }

                //Run Calculations
                computeShader.Dispatch(0, groupCount, 1, 1);

                //Receive Data
                newMatrixValues.GetData(newMat.Values);

                //Get rid of memory
                matrixAValues.Dispose();
                matrixBValues.Dispose();
                newMatrixValues.Dispose();

            } else
                Debug.Log("Error, Dimensions don't match");

            return newMat;
        }

        public static DNAMatrix matrixSubstractionGPU (DNAMatrix matrixA, DNAMatrix matrixB)
        {
            DNAMatrix newMat = new DNAMatrix(0, 0);

            if (_SameDimension(matrixA, matrixB))
            {
                ComputeShader computeShader = matrixSubstractionScript;
                newMat = new DNAMatrix(matrixA.Height, matrixB.Width);

                ComputeBuffer matrixAValues = new ComputeBuffer(matrixA.Values.Length, sizeof(double));
                matrixAValues.SetData(matrixA.Values);

                ComputeBuffer matrixBValues = new ComputeBuffer(matrixB.Values.Length, sizeof(double));
                matrixBValues.SetData(matrixB.Values);

                ComputeBuffer newMatrixValues = new ComputeBuffer(newMat.Values.Length, sizeof(double));
                newMatrixValues.SetData(newMat.Values);

                computeShader.SetBuffer(0, "matrixAValues", matrixAValues);
                computeShader.SetBuffer(0, "matrixBValues", matrixBValues);
                computeShader.SetBuffer(0, "newMatrixValues", newMatrixValues);

                int groupCount = 0;
                if ((newMat.Values.Length / 1024) >= 1)
                    groupCount = (newMat.Values.Length / 1024);
                else
                    groupCount = 1;

                //Run Calculations
                computeShader.Dispatch(0, groupCount, 1, 1);

                //Receive Data
                newMatrixValues.GetData(newMat.Values);

                //Get rid of memory
                matrixAValues.Dispose();
                matrixBValues.Dispose();
                newMatrixValues.Dispose();

            }
            else
                Debug.Log("Error, Dimensions don't match");

            return newMat;
        }

        /// <summary>
        /// Checks if the Matrices are the same Dimension
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        private static bool _SameDimension(DNAMatrix matrixA, DNAMatrix matrixB)
        {
            if (matrixA.Height == matrixB.Height && matrixA.Width == matrixB.Width)
                return true;
            else
                return false;
        }

        /// <summary>
        /// Displays the Matrix in the correct fashion, for debugging purposes
        /// </summary>
        public void DisplayMat()
        {
            //Display the matrix
            string line = "\n";
            for (int height = 0; height < this.Height; height++)
            {
                for (int width = 0; width < this.Width; width++)
                {

                    //Debug.Log("Width: " + width + " Height: " + height + " = " + newMat[getIndex(width, height, dim.x)]);

                    line += this[height, width] + "    ";
                }
                line += "\n";

            }

            Debug.Log(line);
        }
    }
}


using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.Features2D;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace PatternDetection
{
    public class Servicio
    {
        private Dictionary<string, Mat> imageDescriptors = new Dictionary<string, Mat>();

        public void Ejecutar()
        {
            string carpetaMuestras = @"C:\Users\samir\Desktop\Pruebas\Muestras";

            // Verifica si la carpeta de muestras existe
            if (!Directory.Exists(carpetaMuestras))
            {
                Console.WriteLine("La ruta de la carpeta de muestras no es válida.");
                return;
            }

            // Cargar muestras en el diccionario
            CargarMuestras(carpetaMuestras);

            // Comprobar si se cargaron las muestras
            if (imageDescriptors.Count == 0)
            {
                Console.WriteLine("No se encontraron muestras para comparar.");
                return;
            }

            // Captura de video
            using (VideoCapture capture = new VideoCapture(0)) // Ajusta el índice si es necesario
            {
                if (!capture.IsOpened)
                {
                    Console.WriteLine("No se pudo acceder a la cámara.");
                    return;
                }

                Mat frame = new Mat();

                while (true)
                {
                    capture.Read(frame);
                    if (frame.IsEmpty)
                    {
                        Console.WriteLine("No se pudo capturar el fotograma de la cámara.");
                        break;
                    }

                    // Convertir a escala de grises
                    Mat grayFrame = new Mat();
                    CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

                    // Extraer características del fotograma actual
                    Mat currentDescriptor = ExtractFeatures(grayFrame);

                    // Encontrar la coincidencia más cercana
                    string mejorCoincidencia = EncontrarMejorCoincidencia(currentDescriptor);

                    // Mostrar la coincidencia en la consola
                    Console.WriteLine($"Mejor coincidencia para el estampado actual: {mejorCoincidencia}");

                    // Mostrar el fotograma de la cámara
                    CvInvoke.Imshow("Cámara", frame);
                    if (CvInvoke.WaitKey(1) == 27) // Esc para salir
                        break;
                }
            }
            CvInvoke.DestroyAllWindows();
        }

        private void CargarMuestras(string carpetaMuestras)
        {
            string[] imagePaths = Directory.GetFiles(carpetaMuestras, "*.jpg");

            foreach (var imagePath in imagePaths)
            {
                // Obtener el nombre del archivo sin la extensión
                string nombreArchivo = Path.GetFileNameWithoutExtension(imagePath);

                using (Mat image = CvInvoke.Imread(imagePath, ImreadModes.Color))
                {
                    if (image.IsEmpty)
                    {
                        Console.WriteLine($"No se pudo cargar la imagen {imagePath}");
                        continue;
                    }

                    // Convertir a escala de grises
                    Mat grayImage = new Mat();
                    CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);

                    // Extraer características y agregar al diccionario
                    Mat descriptors = ExtractFeatures(grayImage);
                    imageDescriptors[nombreArchivo] = descriptors;

                    Console.WriteLine($"Cargada muestra: {nombreArchivo}");
                }
            }
        }

        private string EncontrarMejorCoincidencia(Mat currentDescriptor)
        {
            var bfMatcher = new BFMatcher(DistanceType.Hamming);
            string mejorCoincidencia = null;
            double menorDistanciaPromedio = double.MaxValue;

            foreach (var muestra in imageDescriptors)
            {
                var matches = new VectorOfDMatch();
                bfMatcher.Match(currentDescriptor, muestra.Value, matches);

                if (matches.ToArray().Length == 0)
                {
                    continue;
                }
                // Calcular distancia promedio
                double avgDistance = matches.ToArray().Average(m => m.Distance);

                // Actualizar la mejor coincidencia si la distancia promedio es menor
                if (avgDistance < menorDistanciaPromedio)
                {
                    menorDistanciaPromedio = avgDistance;
                    mejorCoincidencia = muestra.Key;
                }
            }
            return mejorCoincidencia;
        }

        private Mat ExtractFeatures(Mat grayImage)
        {
            var orb = new ORB();
            var keypoints = new VectorOfKeyPoint();
            var descriptors = new Mat();

            orb.DetectAndCompute(grayImage, null, keypoints, descriptors, false);

            return descriptors;
        }
    }
}

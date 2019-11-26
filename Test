interface Shape {  
   double getArea();
}

class Rectangle implements Shape { 

   private int length;
   private int width;

   public Rectangle(int length, int width) {
      this.length = length;
      this.width = width;
   }

   @Override
   public String toString() {
      return "Rectangle[length=" + length + ",width=" + width + "]";
   }

   @Override
   public double getArea() {
      return length * width;
   }
}

class Triangle implements Shape {

   private int base;
   private int height;

   public Triangle(int base, int height) {
      this.base = base;
      this.height = height;
   }

   @Override
   public String toString() {
      return "Triangle[base=" + base + ",height=" + height + "]";
   }

   @Override
   public double getArea() {
      return 0.5 * base * height;
   }
}

public class Test {
   public static void main(String[] args) {
      Shape s1 = new Rectangle(1, 2); 
      System.out.println(s1);
      System.out.println("Area is " + s1.getArea());

      Shape s2 = new Triangle(3, 4); 
      System.out.println(s2);
      System.out.println("Area is " + s2.getArea());


   }
}

#include<iostream>
#include<vector>

std::vector<std::vector<double>> multiply_matrix(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y)
{
      int x_row = x.size(); int x_col=x[0].size();
      int y_row = y.size(); int y_col = y[0].size();
      std::vector<std::vector<double>> result(x_row,std::vector<double>(y_col));
      if(y_row!=x_col)
      {
           std::cout<<"matrix multiplication not possible"<<std::endl;
      }
      //for row of x
      for(int i=0;i<x_row;i++)
      {
           //for column of y
           for(int k=0;k<y_col;k++)
           {
                //for summation
                for(int l=0;l<x_col;l++)
                {
                     result[i][k]+=(x[i][l]*y[l][k]);
                }
           }
      }
      return result;
}

std::vector<std::vector<double>> add_matrix(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y)
{
      int x_row = x.size(); int x_col=x[0].size();
      int y_row = y.size(); int y_col = y[0].size();
      std::vector<std::vector<double>> result(x_row,std::vector<double>(y_col));
      
      //for row of x
      for(int i=0;i<x_row;i++)
      {
           //for column of y
           for(int k=0;k<y_col;k++)
           {
                result[i][k]=x[i][k]+y[i][k];
           }
      }
      return result;
}

std::vector<std::vector<double>> subtract_matrix(const std::vector<std::vector<double>>& x, const std::vector<std::vector<double>>& y)
{
      int x_row = x.size(); int x_col=x[0].size();
      int y_row = y.size(); int y_col = y[0].size();
      std::vector<std::vector<double>> result(x_row,std::vector<double>(y_col));
      
      //for row of x
      for(int i=0;i<x_row;i++)
      {
           //for column of y
           for(int k=0;k<y_col;k++)
           {
                result[i][k]=x[i][k]-y[i][k];
           }
      }
      return result;
}

std::vector<std::vector<double>> transpose_matrix(const std::vector<std::vector<double>>& x)
{
      int x_row = x.size(); int x_col=x[0].size();
      std::vector<std::vector<double>> result(x_row,std::vector<double>(x_col));
      
      //for row of x
      for(int i=0;i<x_row;i++)
      {
           //for column of y
           for(int k=0;k<x_col;k++)
           {
               result[i][k]=x[k][i];
           }
      }
      return result;
}



/*
int main()
{
      std::vector<std::vector<double>> x = {{1,2,3},{4,5,6},{7,8,9}};
      std::vector<std::vector<double>> y = {{1,2,3},{4,5,6},{7,8,9}};
      std::vector<std::vector<double>> result = transpose_matrix(x);

      for(int i=0;i<result.size();i++)
      {
           for(int k=0;k<3;k++)
           {
                std::cout<<result[i][k]<<" ";
           }
           std::cout<<"\n";
      }
      return 0;
}
*/

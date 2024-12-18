package com.sparrowrecsys.online.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.datamanager.Movie;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

/**
 * MovieService, return information of a specific movie
 */

public class MovieService extends HttpServlet {
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws IOException {
        try {
            response.setContentType("application/json");
            response.setStatus(HttpServletResponse.SC_OK);
            response.setCharacterEncoding("UTF-8");
            response.setHeader("Access-Control-Allow-Origin", "*");

            //get movie id via url parameter
            //从url中获得电影的id
            String movieId = request.getParameter("id");

            //get movie object from DataManager
            //找到datamanager中相同的id1的电影
            Movie movie = DataManager.getInstance().getMovieById(Integer.parseInt(movieId));

            //convert movie object to json format and return
            /**
            *ObjectMapper：
             * 这是 Jackson 库中的核心类，用于将 Java 对象和 JSON 格式之间相互转换。**/

            if (null != movie) {
                ObjectMapper mapper = new ObjectMapper();
                String jsonMovie = mapper.writeValueAsString(movie);
                //System.out.println("movie has found, id = " + movieId);
                //System.out.println("json is "+ jsonMovie);
                response.getWriter().println(jsonMovie);
            }else {
                response.getWriter().println("");
            }

        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println("");
        }
    }
}

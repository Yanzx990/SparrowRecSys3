package com.sparrowrecsys.online.service;
//根据用户 ID、推荐数量、推荐模型生成电影推荐列表，并以 JSON 格式返回给客户端
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sparrowrecsys.online.recprocess.RecForYouProcess;
import com.sparrowrecsys.online.util.ABTest;
import com.sparrowrecsys.online.datamanager.Movie;
import com.sparrowrecsys.online.util.Config;

import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.List;

/**
 * RecForYouService, provide recommended for you service
 */

public class RecForYouService extends HttpServlet {
    protected void doGet(HttpServletRequest request,
                         HttpServletResponse response) throws ServletException,
            IOException {
        try {
            response.setContentType("application/json");
            response.setStatus(HttpServletResponse.SC_OK);
            response.setCharacterEncoding("UTF-8");
            response.setHeader("Access-Control-Allow-Origin", "*");

            //get user id via url parameter
            String userId = request.getParameter("id");
            //number of returned movies
            String size = request.getParameter("size");
            //ranking algorithm
            String model = request.getParameter("model");

            if (Config.IS_ENABLE_AB_TEST){
                model = ABTest.getConfigByUserId(userId);
            }
            //检查系统是否启用了 A/B 测试。
            //A/B 测试是一种实验方法，随机分配用户到不同的模型中，测试哪种模型效果更好。

            //a simple method, just fetch all the movie in the genre
            List<Movie> movies = RecForYouProcess.getRecList(Integer.parseInt(userId), Integer.parseInt(size), model);

            //convert movie list to json format and return
            ObjectMapper mapper = new ObjectMapper();
            String jsonMovies = mapper.writeValueAsString(movies);
            response.getWriter().println(jsonMovies);

        } catch (Exception e) {
            e.printStackTrace();
            response.getWriter().println("");
        }
    }
}

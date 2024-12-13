package com.sparrowrecsys.online;

import com.sparrowrecsys.online.datamanager.DataManager;
import com.sparrowrecsys.online.service.*;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.DefaultServlet;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.eclipse.jetty.util.resource.Resource;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URL;

/***
 * Recsys Server, end point of online recommendation service
 */

public class RecSysServer {

    public static void main(String[] args) throws Exception {
        new RecSysServer().run();
        //调用了这个类的run方法
    }

    //recsys server port number
    private static final int DEFAULT_PORT = 6010;

    public void run() throws Exception{

        int port = DEFAULT_PORT;
        try {
            port = Integer.parseInt(System.getenv("PORT"));
        } catch (NumberFormatException ignored) {}

        //set ip and port number
        //这行代码的作用是创建一个 InetSocketAddress 对象 inetAddress，它代表了服务器的 IP 地址和端口组合。在这个例子中，服务器将绑定到 0.0.0.0 地址和 port 端口。
        InetSocketAddress inetAddress = new InetSocketAddress("0.0.0.0", port);
        Server server = new Server(inetAddress);

        //get index.html path
        URL webRootLocation = this.getClass().getResource("/webroot/index.html");
        if (webRootLocation == null)
        {
            throw new IllegalStateException("Unable to determine webroot URL location");
        }

        //set index.html as the root page
        URI webRootUri = URI.create(webRootLocation.toURI().toASCIIString().replaceFirst("/index.html$","/"));
        System.out.printf("Web Root URI: %s%n", webRootUri.getPath());

        //load all the data to DataManager
        //CSV文件是一种文本文件，可以用来存储数据
        DataManager.getInstance().loadData(webRootUri.getPath() + "sampledata/movies.csv",
                webRootUri.getPath() + "sampledata/links.csv",webRootUri.getPath() + "sampledata/ratings.csv",
                webRootUri.getPath() + "modeldata/item2vecEmb.csv",
                webRootUri.getPath() + "modeldata/userEmb.csv",
                "i2vEmb", "uEmb");

        //create server context
        //用于管理 HTTP 服务器的上下文配置，包括路径、资源、欢迎页面等。
        ServletContextHandler context = new ServletContextHandler();
        //设置服务器的上下文路径为根路径 /，表示所有请求都从根路径开始处理。
        context.setContextPath("/");
        //将服务器的根目录设置为 webRootUri 指定的目录（例如 /webroot/）。
        //服务器会从该目录中加载静态资源。
        context.setBaseResource(Resource.newResource(webRootUri));
        context.setWelcomeFiles(new String[] { "index.html" });
        //添加 MIME 类型映射，这里将 .txt 文件的 MIME 类型设置为 text/plain;charset=utf-8，以支持 UTF-8 编码的纯文本文件。
        context.getMimeTypes().addMimeMapping("txt","text/plain;charset=utf-8");

        //bind services with different servlets
        context.addServlet(DefaultServlet.class,"/");
        context.addServlet(new ServletHolder(new MovieService()), "/getmovie");
        context.addServlet(new ServletHolder(new UserService()), "/getuser");
        context.addServlet(new ServletHolder(new SimilarMovieService()), "/getsimilarmovie");
        context.addServlet(new ServletHolder(new RecommendationService()), "/getrecommendation");
        context.addServlet(new ServletHolder(new RecForYouService()), "/getrecforyou");
        context.addServlet(new ServletHolder(new SearchMovieService()), "/searchmovie");
        //set url handler
        server.setHandler(context);
        System.out.println("RecSys Server has started.");

        //start Server
        server.start();
        server.join();
    }
}
